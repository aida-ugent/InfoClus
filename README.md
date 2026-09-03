# InfoClus

InfoClus is an interactive dashboard for **explainable clustering**: it clusters your dataset using a hierarchical, information-theoretic method, projects it into 2D for visual exploration, and — for every cluster it finds — surfaces the attributes that best explain why the cluster exists.

Built with [Dash](https://dash.plotly.com/) / [Plotly](https://plotly.com/python/), developed at [AIDA, Ghent University](https://aida.ugent.be/).

## Features

- Hierarchical clustering with an interactive dendrogram, tunable at runtime (alpha, beta, min/max number of explanatory attributes)
- 2D embedding view (t-SNE / PCA) synced with the cluster explanations
- Per-cluster attribute explanations, ranked by how well they characterize the cluster
- Import your own dataset or embedding directly from the dashboard
- Optional LLM-assisted cluster labeling (bring your own API key)
- Ships with three ready-to-explore datasets, embeddings pre-computed

## Getting started

**Requirements:** Python 3.10+ (developed and tested on 3.14).

```bash
pip install -r requirements.txt
```

**Run the dashboard:**

```bash
cd src
python app.py
```

Then open http://127.0.0.1:8051 in your browser. The dashboard starts on the `cytometry_2500` dataset by default; use the **Dataset** dropdown to switch to any of the other bundled datasets.

## Included datasets

| Dataset | Description | Rows | Attributes |
|---|---|---|---|
| `cytometry_2500` | Flow cytometry marker measurements (2,500-cell subsample) | 2,500 | 9 |
| `german_socio_eco` | Socio-economic indicators for German administrative regions | 412 | 31 |
| `ImmuneW_HDdata` | Whole-blood immune cell profiling data | 254,497 | 36 |

Each dataset ships with a pre-computed `embeddings.npz` (t-SNE and/or PCA), so the dashboard opens instantly. The hierarchical clustering itself is **not** pre-cached — the first time you open a dataset in a session, InfoClus will compute it on the fly, then reuse the result for the rest of that session. `ImmuneW_HDdata` has ~250K rows, so expect the first load of that dataset in particular to take noticeably longer than the other two.

## Using your own dataset

1. Create a new folder under `data/` named after your dataset, e.g. `data/my_dataset/`.
2. Put your data as `data/my_dataset/my_dataset.csv` (rows = instances, columns = attributes).
3. Restart the dashboard, or use the **Upload Dataset** button in the sidebar.

InfoClus will compute embeddings (t-SNE and PCA) and cache them to `embeddings.npz` automatically on first load. You can also drop in a pre-computed embedding via the **Upload Embedding** button.

## Optional: LLM-assisted cluster labeling

The dashboard has an optional panel that asks a large language model to suggest a human-readable label for a cluster, given its explanatory attributes. This is **fully optional** and disabled by default — it only runs when you supply your own API key in the panel. Supported providers: OpenAI, Anthropic, Google, or a custom OpenAI-compatible endpoint. No key is stored anywhere in this repository; you paste it directly into the running dashboard.

## Project structure

```
InfoClus-release/
├── src/
│   ├── app.py              # Dash app entry point (run this)
│   ├── infoclus.py         # Core InfoClus clustering & explanation algorithm
│   ├── infoclus_utils.py   # Data preprocessing, embeddings, helper functions
│   ├── dash_utils.py       # Dashboard-side helpers (caching, import/export, LLM calls)
│   ├── layout.py           # Dashboard layout
│   ├── callbacks.py        # Dashboard interactivity
│   ├── caching.py          # Simple pickle-based cache
│   ├── config.py           # Paths and global settings
│   └── assets/             # CSS
├── data/
│   ├── cytometry_2500/
│   ├── german_socio_eco/
│   └── ImmuneW_HDdata/
├── requirements.txt
└── LICENSE
```

## License

MIT — see [LICENSE](LICENSE).
