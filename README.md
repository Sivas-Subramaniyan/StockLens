# StockLens — AI Research Terminal

An AI-powered stock research tool for Indian smallcap equities. It ranks companies by fundamental quality, then runs a fully-automated deep-research workflow — web searches, full page fetching, Gemini-generated analyst reports, and a BUY / AVOID verdict — all through a clean browser UI.

---

## Features

- **Fundamental scoring** — 8-metric weighted algorithm (ROCE, FCF, working capital, debt/equity, valuation, OPM, promoter holding) ranks the entire screener universe
- **12-category deep research** — 40+ DuckDuckGo searches per company across business fundamentals, financial health, moat, management, risks, governance, and more
- **AI-optimised queries** — one Gemini call at the start generates targeted search queries for every subtopic instead of plain keyword templates
- **Full page content** — top 2 URLs per query are fetched and parsed in parallel so Gemini sees actual article text, not just snippets
- **Research caching** — already-searched categories are skipped on same-day re-runs; a full cache hit skips query generation entirely
- **Gemini analyst report** — comprehensive markdown report covering all 12 research categories
- **Combined validation + TLDR** — a single Gemini JSON call returns the BUY/AVOID verdict, confidence, 3-year return estimate, key drivers/risks, red flags, and an executive summary paragraph
- **Persistent job store** — completed research jobs survive server restarts
- **Runtime API key** — paste your Gemini key in the browser; no server restart needed
- **Render-ready** — `render.yaml` and `Procfile` included for one-click cloud deployment

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn (Python 3.11) |
| AI / LLM | Google Gemini 2.5 Flash (free tier) |
| Web search | DuckDuckGo (`ddgs`) — no API key required |
| Frontend | Vanilla HTML / CSS / JS, marked.js |
| Data | Screener.in export → `ranked_companies.csv` |
| Deployment | Render (PaaS) |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Sivas-Subramaniyan/StockLens.git
cd StockLens
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Get a free Gemini API key

Grab one from [Google AI Studio](https://aistudio.google.com/apikey) — it's free, no credit card needed.

```bash
# Option A — environment variable (recommended for production)
export GEMINI_API_KEY=AIza...

# Option B — paste it in the browser after the server starts (easiest)
```

### 3. Run

```bash
python run_api.py
```

Open `http://localhost:8000`. If no key is configured via env var, the **Research** tab shows a yellow banner where you can paste and activate your key directly in the browser.

---

## Workflow

```
Browser
  │
  ▼
FastAPI  (api.py)
  │
  ├─ CompanySelector      reads ranked_companies.csv
  │
  ├─ ResearchAgent        (research_agent.py)
  │    ├─ Gemini: generate optimised queries   [1 call — skipped if all cached]
  │    ├─ DuckDuckGo: ~42 searches, 10 results each
  │    └─ requests: parallel page fetch for top-2 URLs per query
  │
  └─ SummarizationAgent   (summarization_agent.py)
       ├─ Gemini: full analyst report           [1 call, markdown]
       └─ Gemini: BUY/AVOID validation + TLDR  [1 combined JSON call]

Total Gemini calls per fresh session : 3
Total Gemini calls on full cache hit  : 2
```

---

## Scoring Algorithm

Eight metrics, winsorized (5th–95th percentile) then min-max normalised:

| Metric | Weight | Direction |
|---|---|---|
| ROCE | 20% | Higher is better |
| FCF (3Y) | 20% | Higher is better |
| Working capital efficiency | 15% | Lower is better |
| Debt / Equity | 10% | Lower is better |
| Valuation (P/E vs industry) | 10% | Lower is better |
| Operating cash flow (3Y) | 10% | Higher is better |
| Operating profit margin | 10% | Higher is better |
| Promoter holding | 5% | Higher is better |

Run `python score_companies.py` to regenerate `ranked_companies.csv` from a fresh Screener export.

---

## Deploying to Render

1. Push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com) and connect the repo
3. Render auto-detects `render.yaml` — build and start commands are pre-configured
4. Add the environment variable `GEMINI_API_KEY` in the Render dashboard
5. Deploy — done

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes (or set in UI) | Google AI Studio API key |
| `API_HOST` | No | Bind address (default `0.0.0.0`) |
| `API_PORT` | No | Port (default `8000`) |

---

## Project Structure

```
StockLens/
├── api.py                   # FastAPI app — routes, workflow orchestration
├── research_agent.py        # DuckDuckGo search + AI query gen + page fetching
├── summarization_agent.py   # Gemini report, validation + TLDR
├── company_selector.py      # Load and query ranked_companies.csv
├── score_companies.py       # Scoring algorithm (run to rebuild ranked_companies.csv)
├── research_orchestrator.py # LangGraph-based alternative workflow
├── run_api.py               # Server entry point (uvicorn with auto-reload)
├── ranked_companies.csv     # Pre-scored company universe
├── requirements.txt
├── render.yaml              # Render PaaS deployment config
├── Procfile                 # Heroku / Railway / Fly.io start command
└── frontend/
    ├── index.html
    ├── script.js
    └── styles.css
```

---

## License

MIT
