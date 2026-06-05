# KOL Atlas — Key Opinion Leader Intelligence

A mini data science project that extracts, compares, and ranks medical researchers
using public profile data, AI embeddings, and a custom influence scoring model.

## Project Structure

| File | Description |
|------|-------------|
| `kol_atlas.ipynb` | Main notebook with full pipeline |
| `app.py` | Streamlit web app |
| `kol_output.json` | Extracted KOL data |
| `similarity_heatmap.png` | Embedding similarity heatmap |
| `kol_clusters.png` | KOL clustering visualization |
| `requirements.txt` | Dependencies |
| `README.md` | Project documentation |


## What This Project Does

- Extracts structured data from 3 medical researchers via Semantic Scholar API
- Generates AI embeddings using sentence-transformers
- Computes cosine similarity between KOL profiles
- Ranks KOLs using a custom 5-factor influence score
- Clusters KOLs using KMeans + PCA
- Compares two KOLs using LLM (Groq / Llama 3.3)

---

## KOLs Selected

| Name | Field | Affiliation |
|------|-------|-------------|
| Eric Topol | Cardiology + AI in Medicine | Scripps Research Translational Institute |
| Siddhartha Mukherjee | Oncology | Columbia University |
| David Sinclair | Aging + Longevity Biology | Harvard Medical School |

---

## Influence Score Formula

| Factor | Weight |
|--------|--------|
| H-Index | 25% |
| Total Citations | 25% |
| Total Publications | 20% |
| Recent Publications (last 3 years) | 20% |
| Topic Diversity | 10% |

---

## How to Run

### Notebook
Open `kol_atlas.ipynb` in Jupyter or Google Colab and run all cells.

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Optional Add-Ons Completed

- ✅ Confidence scores on extracted fields
- ✅ KOL clustering using embeddings
- ✅ LLM-powered KOL comparison (Groq)
- ✅ Streamlit UI

---

## Data Source

All data fetched from [Semantic Scholar API](https://api.semanticscholar.org/) —
free, no authentication required.

