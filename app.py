import time
import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# ── Page Config ──────────────────────────────────────────
st.set_page_config(page_title="KOL Atlas", page_icon="🔬", layout="wide")
st.title("🔬 KOL Atlas — Key Opinion Leader Intelligence")
st.markdown("Extract, compare, and rank medical researchers using AI.")

# ── KOL Definitions ──────────────────────────────────────
KOL_URLS = [
    {"name": "Eric Topol", "field": "Cardiology + AI in Medicine",
     "authorId": "144758045", "affiliation": "Scripps Research Translational Institute",
     "topics": ["Artificial Intelligence", "Cardiology", "Genomics", "Digital Medicine"],
     "url": "https://scholar.google.com/citations?user=E2-uIQYAAAAJ"},
    {"name": "Siddhartha Mukherjee", "field": "Oncology",
     "authorId": "49581145", "affiliation": "Columbia University",
     "topics": ["Oncology", "Cancer Biology", "Hematology", "Stem Cell Biology"],
     "url": "https://scholar.google.com/citations?user=auZtXcQAAAAJ"},
    {"name": "David Sinclair", "field": "Aging + Longevity Biology",
     "authorId": "11152645", "affiliation": "Harvard Medical School",
     "topics": ["Aging", "Epigenetics", "Longevity", "Sirtuins", "NAD+ Metabolism"],
     "url": "https://scholar.google.com/citations?hl=en&user=lfLudSQAAAAJ"},
]

# ── Data Extraction ───────────────────────────────────────
@st.cache_data
def load_kol_data():
    kol_data = []
    for kol_info in KOL_URLS:
        time.sleep(2)
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/author/{kol_info['authorId']}",
            params={"fields": "name,affiliations,hIndex,citationCount,paperCount,papers.year"},
            timeout=10
        )
        data = response.json()
        current_year = datetime.now().year
        recent_pubs = sum(
            1 for p in data.get("papers", [])
            if str(p.get("year", "")).isdigit() and current_year - int(p["year"]) <= 3
        )
        kol_data.append({
            "name":                kol_info["name"],
            "field":               kol_info["field"],
            "affiliation":         kol_info.get("affiliation", "N/A"),
            "h_index":             data.get("hIndex", 0),
            "total_citations":     data.get("citationCount", 0),
            "total_publications":  data.get("paperCount", 0),
            "recent_publications": recent_pubs,
            "research_topics":     kol_info["topics"],
            "topic_diversity":     len(kol_info["topics"]),
            "profile_source":      "Semantic Scholar",
            "profile_url":         kol_info["url"],
            "confidence_score":    0.95
        })
    return kol_data

# ── Influence Score ───────────────────────────────────────
def calculate_influence_score(kol):
    return round(
        (0.25 * kol["h_index"]) +
        (0.25 * (kol["total_citations"] / 100)) +
        (0.20 * kol["total_publications"]) +
        (0.20 * kol["recent_publications"]) +
        (0.10 * kol["topic_diversity"]), 2
    )

# ── Load Data ─────────────────────────────────────────────
with st.spinner("Fetching KOL data from Semantic Scholar..."):
    kol_data = load_kol_data()

for k in kol_data:
    k["influence_score"] = calculate_influence_score(k)

# ── Section 1: KOL Profiles ───────────────────────────────
st.header("📋 KOL Profiles")
cols = st.columns(3)
for i, kol in enumerate(kol_data):
    with cols[i]:
        st.subheader(kol["name"])
        st.caption(kol["field"])
        st.write(f"🏛️ {kol['affiliation']}")
        st.metric("H-Index", kol["h_index"])
        st.metric("Citations", f"{kol['total_citations']:,}")
        st.metric("Publications", kol["total_publications"])
        st.metric("Influence Score", kol["influence_score"])
        st.write("**Topics:**", ", ".join(kol["research_topics"]))

# ── Section 2: Influence Score Table ─────────────────────
st.header("🏆 Influence Score Ranking")
scores_df = pd.DataFrame([{
    "Name": k["name"],
    "Field": k["field"],
    "H-Index": k["h_index"],
    "Citations": k["total_citations"],
    "Publications": k["total_publications"],
    "Recent Pubs": k["recent_publications"],
    "Topic Diversity": k["topic_diversity"],
    "Influence Score": k["influence_score"]
} for k in kol_data]).sort_values("Influence Score", ascending=False).reset_index(drop=True)

st.dataframe(scores_df, use_container_width=True)

# ── Section 3: Similarity Heatmap ────────────────────────
st.header("🔁 Embedding Similarity Matrix")

@st.cache_resource
def get_embeddings(kol_data):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [
        f"{k['name']} works in {k['field']} at {k['affiliation']}. "
        f"Research: {', '.join(k['research_topics'])}. "
        f"H-index: {k['h_index']}, Citations: {k['total_citations']}."
        for k in kol_data
    ]
    return model.encode(texts)

embeddings = get_embeddings(kol_data)
names = [k["name"] for k in kol_data]
sim_matrix = cosine_similarity(embeddings)
sim_df = pd.DataFrame(sim_matrix, index=names, columns=names).round(3)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(sim_df, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, vmin=0, vmax=1, ax=ax)
ax.set_title("KOL Embedding Similarity", fontweight="bold")
plt.tight_layout()
st.pyplot(fig)

# ── Section 4: Clustering ─────────────────────────────────
st.header("🧬 KOL Clustering")

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

fig2, ax2 = plt.subplots(figsize=(6, 4))
colors = ["#e63946", "#2a9d8f", "#e9c46a"]
for i, (name, coord) in enumerate(zip(names, reduced)):
    ax2.scatter(coord[0], coord[1], color=colors[i], s=200, zorder=3)
    ax2.annotate(name, (coord[0], coord[1]),
                 textcoords="offset points", xytext=(10, 5), fontsize=10)
ax2.set_title("KOL Clustering (PCA + KMeans)", fontweight="bold")
plt.tight_layout()
st.pyplot(fig2)

# ── Section 5: LLM Comparison ────────────────────────────
st.header("🤖 LLM-Powered KOL Comparison")

groq_key = st.text_input("Enter your Groq API Key", type="password")
kol_names = [k["name"] for k in kol_data]
col1, col2 = st.columns(2)
with col1:
    kol1_name = st.selectbox("Select KOL 1", kol_names, index=0)
with col2:
    kol2_name = st.selectbox("Select KOL 2", kol_names, index=2)

if st.button("Compare KOLs"):
    if not groq_key:
        st.warning("Please enter your Groq API key above.")
    else:
        kol1 = next(k for k in kol_data if k["name"] == kol1_name)
        kol2 = next(k for k in kol_data if k["name"] == kol2_name)
        with st.spinner("Generating comparison..."):
            client_llm = Groq(api_key=groq_key)
            prompt = f"""Compare these two researchers professionally in 3-4 lines:
KOL 1: {kol1['name']} | Field: {kol1['field']} | H-Index: {kol1['h_index']} | Citations: {kol1['total_citations']} | Topics: {', '.join(kol1['research_topics'])}
KOL 2: {kol2['name']} | Field: {kol2['field']} | H-Index: {kol2['h_index']} | Citations: {kol2['total_citations']} | Topics: {', '.join(kol2['research_topics'])}"""
            response = client_llm.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            st.success(response.choices[0].message.content)
