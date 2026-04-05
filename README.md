# Data-Scraping-Agent

An AI-powered local business intelligence agent that answers the question: **"Is this area a good location to open or compete in this type of business?"**

Provide a natural-language query like `cafes near Park Street Kolkata`. The agent scrapes Google Places, scores every result against a parameterized suitability model, runs each through a local LLM for a strategic verdict, and exports a ranked CSV plus research artifacts to `~/Downloads` (or `outputs/` when `~/Downloads` is unavailable).

---

## How it works

```
Query (e.g. "restaurants near Connaught Place Delhi")
    │
    ▼
Google Places API v1 (text search, up to 3 pages)
    │
    ▼
Normalize + deduplicate results (name + ≤50m lat/lng tolerance)
    │
    ▼
Haversine competitor analysis (configurable radius, default 500m)
    │
    ▼
Scoring model  →  rating + review volume + competitor density + competitor quality (parameterized weights)
    │
    ▼
Flan-T5 verdict  →  pros / cons / next steps per business
    │
    ▼
Timestamped CSV  →  ~/Downloads/suitability_results_<timestamp>.csv
```

---

## Scoring rubric

| Signal | Condition | Points |
|---|---|---|
| Rating | ≥ 4.5 | +3 |
| | ≥ 4.0 | +2 |
| | ≥ 3.5 | +1 |
| Review count | ≥ 2000 | +3 |
| | ≥ 700 | +2 |
| | ≥ 150 | +1 |
| Competitor density | ≤ 2 nearby | +2 |
| | ≤ 5 nearby | +1 |
| | > 5 nearby | −1 |
| Competitor strength | avg rating < 3.9 | +1 |
| | avg rating ≥ 4.3 | −1 |

**Labels**: Highly suitable (≥7) · Moderately suitable (≥4) · Not recommended (<4)

---

## Output CSV columns

`Business Name · Type · Rating · Total Reviews · Address · Lat · Lng · Local Competitors · Avg Local Comp Rating · Score · Label · Verdict · Moran I Approx · Competitor HHI`

---

## Setup

**Prerequisites**: Python 3.9+, a Google Cloud project with the Places API (New) enabled.

```bash
git clone https://github.com/td-02/Data-Scraping-Agent.git
cd Data-Scraping-Agent
pip install python-dotenv requests pandas scipy transformers torch tqdm
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_places_api_key
HF_API_TOKEN=your_huggingface_token   # optional — only needed for private models
```

---

## Usage

### CLI (recommended)

```bash
python business_suitability_single_file.py
python business_suitability_single_file.py --seed 42   # optional, fixes any stochasticity in the scoring / analysis
```

```
Enter business query (e.g. 'cafes near Park Street Kolkata'): restaurants near Bandra Mumbai
Pages to fetch (1-3, default 2): 2
Local radius in meters for 'local competitors' (default 500): 500
```

Every run writes `suitability_results_<timestamp>.csv`, a corresponding `_run_manifest.json`, and a `sensitivity_results_<timestamp>.csv` (all under `~/Downloads` or `outputs/`). The CLI also prints Moran's I and a baseline Kendall tau table in the terminal so you can cite geographic clustering and rating/review correlations.

### Streamlit UI

```bash
streamlit run app.py
```

Enter a prompt, click **Run Agent**, preview the top 20 results, and download the full CSV.

---

## Research instrumentation

- **Parameterized viability model**: `compute_business_score()` exposes weights for rating / reviews / competition / competitor quality, and you can study perturbations via the released `sensitivity_results` CSV.
- **Reproducible config**: every run logs `*_run_manifest.json` alongside the CSV, recording `query`, `pages`, `radius_m`, `weights`, `seed`, `model_name`, `Moran's I`, and the linked CSV artifacts.
- **Analysis outputs**: the CLI prints Moran's I (spatial autocorrelation) plus a baseline correlation table (Kendall's tau vs. rating-only and review-count-only rankings). The sensitivity DataFrame is provided by `sensitivity.py`, and `baseline.py` exposes the Kendall computation separately if you want to reuse it offline.

## Project structure

```
Data-Scraping-Agent/
├── business_suitability_single_file.py   # main pipeline (use this)
├── business_agent.py                     # v2 — Places API v1, BART summarizer
├── bizagent_api.py                       # v1 — legacy googlemaps SDK
├── baseline.py                           # Kendall tau baseline comparison
├── sensitivity.py                        # weight-perturbation sensitivity sweep
├── app.py                                # Streamlit frontend
├── requirements.txt
└── .env                                  # not committed — add your own
```

The three Python files represent the evolution of the agent. `business_suitability_single_file.py` is the current, fully-featured version — it adds deduplication, Haversine competitor radius, the suitability rubric, and Flan-T5 reasoning over the earlier prototypes.

---

## Tech stack

- **Google Places API v1** — text search with field masks
- **Haversine formula** — geographic competitor radius calculation
- **Flan-T5-base** (HuggingFace Transformers) — local LLM, runs on CPU
- **tqdm** — progress bar over the evaluation loop
- **Streamlit** — optional web UI
- **python-dotenv** — API key management

---

## Limitations

- The LLM verdict quality is constrained by Flan-T5's capacity — it generates plausible but generic business advice. Swapping to a larger model (e.g. via the Anthropic or OpenAI API) would improve output significantly.
- The competitor pool is limited to whatever the Places API returns for the query, not all businesses in the area.
- No result caching — each run re-hits the Places API and incurs quota usage.

---

## License

MIT
