"""
FastAPI backend for Stock Research Tool
Gemini-powered, free-tier optimised.
Supports runtime API-key update and token-usage tracking.

Improvements:
  - Gap 4: TLDR is captured from save_report() and returned in /research/results/:id
  - Gap 5: Completed/errored jobs are persisted to research_jobs_store.json
           and reloaded on server start so they survive restarts.
  - ResearchAgent now receives the Gemini key so it can generate AI queries.
  - Hardcoded API key fallback removed — use the GEMINI_API_KEY env var or
    the frontend Settings drawer to set the key at runtime.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime
import asyncio
import threading

# Fix Windows encoding
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

from company_selector import CompanySelector
from research_agent import ResearchAgent
from summarization_agent import SummarizationAgent

app = FastAPI(title="Stock Research Tool API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# ── Mutable global state ───────────────────────────────────────────────────
research_jobs: Dict[str, Dict] = {}

# API key — mutable so the frontend can update it at runtime.
# No hardcoded fallback — users must supply their key via env var or the UI.
_config: Dict = {
    "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
    "model":          "models/gemini-2.5-flash",
    "key_status":     "unknown",   # "active" | "invalid" | "unknown"
    "key_label":      "",          # masked display e.g. "AIza…SxI"
}

# Cumulative token usage across all sessions
_token_stats: Dict = {
    "input_tokens":  0,
    "output_tokens": 0,
    "total_tokens":  0,
    "api_calls":     0,
    "sessions":      0,
    "last_updated":  None,
}

RESEARCH_OUTPUT_DIR = "research_output"
REPORTS_DIR         = "reports"

# ── Gap 5: Persistent job store ────────────────────────────────────────────
JOBS_STORE_FILE = "research_jobs_store.json"

# Lazy singleton
_company_selector: Optional[CompanySelector] = None


# ── Helpers ────────────────────────────────────────────────────────────────

def _mask_key(key: str) -> str:
    if not key or len(key) < 8:
        return "***"
    return key[:4] + "…" + key[-4:]


def get_company_selector() -> CompanySelector:
    global _company_selector
    if _company_selector is None:
        _company_selector = CompanySelector()
    return _company_selector


def get_gemini_key() -> str:
    return _config["gemini_api_key"]


def _accumulate_tokens(agent: SummarizationAgent):
    """Add a finished agent's token stats to the global counter."""
    stats = agent.token_stats
    _token_stats["input_tokens"]  += stats.get("input_tokens", 0)
    _token_stats["output_tokens"] += stats.get("output_tokens", 0)
    _token_stats["total_tokens"]  += stats.get("total_tokens", 0)
    _token_stats["api_calls"]     += stats.get("api_calls", 0)
    _token_stats["sessions"]      += 1
    _token_stats["last_updated"]   = datetime.now().isoformat()


# ── Gap 5: Job persistence helpers ────────────────────────────────────────

def _load_jobs_from_store():
    """Load previously completed/errored jobs from disk into memory."""
    store = Path(JOBS_STORE_FILE)
    if not store.exists():
        return
    try:
        with open(store, "r", encoding="utf-8") as f:
            saved: Dict = json.load(f)
        loaded = 0
        for job_id, job in saved.items():
            if job_id not in research_jobs:  # don't overwrite active jobs
                research_jobs[job_id] = job
                loaded += 1
        if loaded:
            print(f"[JobStore] Loaded {loaded} historical jobs from {JOBS_STORE_FILE}")
    except Exception as e:
        print(f"[JobStore] Could not load store: {e}")


def _save_job_to_store(job: Dict):
    """Append / update a single job in the on-disk store."""
    store = Path(JOBS_STORE_FILE)
    try:
        existing: Dict = {}
        if store.exists():
            with open(store, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing[job["research_id"]] = job
        with open(store, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[JobStore] Could not save job: {e}")


# Load historical jobs when the module is imported
_load_jobs_from_store()


# ── Pydantic models ────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    company_name: str
    company_rank: Optional[int] = None

class ResearchStatus(BaseModel):
    research_id:  str
    status:       str
    company_name: str
    progress:     Dict
    current_step: str
    error:        Optional[str] = None

class CompanyInfo(BaseModel):
    rank:             int
    name:             str
    market_cap:       str
    pe_ratio:         str
    roce:             str
    investment_score: str

class UpdateApiKeyRequest(BaseModel):
    api_key: str


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    frontend_file = Path(__file__).parent / "frontend" / "index.html"
    if frontend_file.exists():
        with open(frontend_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"message": "Stock Research Tool API v2.1 (Gemini)", "status": "running"}


@app.get("/api")
async def api_info():
    return {
        "message":    "Stock Research Tool API v2.1",
        "ai_backend": "Google Gemini (free tier)",
        "search":     "DuckDuckGo (free)",
        "features": [
            "AI-generated search queries (gap 1)",
            "Full page content fetching (gap 2)",
            "Per-category research caching (gap 3)",
            "TLDR in API results (gap 4)",
            "Persistent job store (gap 5)",
        ],
    }


@app.get("/companies", response_model=List[CompanyInfo])
async def get_companies(top_n: Optional[int] = None):
    try:
        df = get_company_selector().list_companies(top_n=top_n)
        return [
            CompanyInfo(
                rank=int(row["rank"]),
                name=str(row["Name"]),
                market_cap=str(row.get("Mar Cap Rs.Cr.", "N/A")),
                pe_ratio=str(row.get("P/E", "N/A")),
                roce=str(row.get("ROCE %", "N/A")),
                investment_score=str(row.get("investment_score", "N/A")),
            )
            for _, row in df.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scoring-algorithm")
async def get_scoring_algorithm():
    return {
        "weights": {
            "roce":          0.20, "fcf_3y":       0.20, "wc_efficiency": 0.15,
            "debt_eq":       0.10, "valuation":     0.10, "cf_op_3y":      0.10,
            "opm":           0.10, "prom_hold":     0.05,
        },
        "description": (
            "Weighted scoring system across 8 financial metrics. "
            "Each metric is winsorized and normalised to 0-1 before weighting."
        ),
        "metrics": {
            "roce":          "Return on Capital Employed — Higher is better (20%)",
            "fcf_3y":        "Free Cash Flow over 3 years — Higher is better (20%)",
            "wc_efficiency": "Working Capital Efficiency — Lower is better, inverted (15%)",
            "debt_eq":       "Debt/Equity ratio — Lower is better, inverted (10%)",
            "valuation":     "P/E vs Industry P/E — Lower is better, inverted (10%)",
            "cf_op_3y":      "Operating Cash Flow 3Y — Higher is better (10%)",
            "opm":           "Operating Profit Margin — Higher is better (10%)",
            "prom_hold":     "Promoter Holding % — Higher is better (5%)",
        },
        "process": [
            "1. Load financial data from screener results",
            "2. Winsorise each metric (5th–95th percentile)",
            "3. Normalise to 0-1 scale (min-max)",
            "4. Invert metrics where lower is better",
            "5. Calculate weighted sum",
            "6. Rank companies by investment score",
        ],
    }


# ── Settings endpoints ─────────────────────────────────────────────────────

@app.get("/settings")
async def get_settings():
    """Return current settings (key masked)."""
    key = _config["gemini_api_key"]
    status = _config["key_status"]
    # Auto-detect unknown status if a key is present
    if key and status == "unknown":
        status = "unknown"  # stays unknown until a real verify call
    return {
        "key_label":  _mask_key(key),
        "key_status": status,
        "model":      _config["model"],
        "has_key":    bool(key),
    }


@app.post("/settings/api-key")
async def update_api_key(req: UpdateApiKeyRequest):
    """
    Update the Gemini API key at runtime.
    Validates the new key with a quick test call.
    Resets all cached state so the new key takes effect immediately.
    """
    global _company_selector

    new_key = req.api_key.strip()
    if not new_key:
        raise HTTPException(status_code=400, detail="API key cannot be empty")

    # ── Test the key ───────────────────────────────────────────────────────
    try:
        from google import genai
        from google.genai import types
        client   = genai.Client(api_key=new_key)
        test_cfg = types.GenerateContentConfig(max_output_tokens=16, temperature=0.0)
        resp     = client.models.generate_content(
            model=_config["model"],
            contents="Reply with exactly one word: ready",
            config=test_cfg,
        )
        text = resp.text or ""
        if text is None and resp.candidates:
            parts = resp.candidates[0].content.parts or []
            text  = "".join(p.text for p in parts if getattr(p, "text", None)) or ""
    except Exception as e:
        err = str(e)
        _config["key_status"] = "invalid"
        status_code = 402 if "429" in err or "quota" in err.lower() else 400
        raise HTTPException(
            status_code=status_code,
            detail=f"Key validation failed: {err[:200]}"
        )

    # ── Accept the key ─────────────────────────────────────────────────────
    _config["gemini_api_key"] = new_key
    _config["key_status"]     = "active"
    _config["key_label"]      = _mask_key(new_key)

    # Clear cached singleton so the next request picks up the new key
    _company_selector = None

    return {
        "status":    "active",
        "key_label": _mask_key(new_key),
        "model":     _config["model"],
        "message":   "API key verified and applied. System ready.",
    }


# ── Token stats endpoint ───────────────────────────────────────────────────

@app.get("/stats/tokens")
async def get_token_stats():
    """Return cumulative token usage across all research sessions."""
    return {
        **_token_stats,
        "key_label":  _mask_key(_config["gemini_api_key"]),
        "key_status": _config["key_status"],
        "has_key":    bool(_config["gemini_api_key"]),
    }


# ── Research endpoints ─────────────────────────────────────────────────────

@app.post("/research/start")
async def start_research(request: ResearchRequest):
    # Guard: must have an API key
    if not _config["gemini_api_key"]:
        raise HTTPException(
            status_code=400,
            detail="No Gemini API key configured. Please set it in Settings before starting research.",
        )

    try:
        selector     = get_company_selector()
        company_data = (
            selector.get_company_by_rank(request.company_rank)
            if request.company_rank
            else selector.get_company_by_name(request.company_name)
        )
        if not company_data:
            raise HTTPException(status_code=404, detail=f"Company not found: {request.company_name}")

        research_id = str(uuid.uuid4())
        research_jobs[research_id] = {
            "research_id":    research_id,
            "status":         "started",
            "company_name":   company_data["company_name"],
            "financial_data": company_data["financial_data"],
            "progress": {
                "step":    "Initialising",
                "current": 0,
                "total":   4,
                "message": "Starting research workflow…",
            },
            "current_step": "initialising",
            "error":        None,
            "started_at":   datetime.now().isoformat(),
            "results":      None,
        }

        t = threading.Thread(target=_run_sync, args=(research_id, company_data))
        t.daemon = True
        t.start()

        return {
            "research_id":  research_id,
            "status":       "started",
            "company_name": company_data["company_name"],
            "message":      "Research started",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_sync(research_id: str, company_data: Dict):
    import traceback
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_workflow(research_id, company_data))
    except Exception as e:
        details = traceback.format_exc()
        job = research_jobs.get(research_id, {})
        job.update({
            "status":        "error",
            "error":         str(e),
            "error_details": details,
            "current_step":  "error",
            "progress": {
                "step":    "Error",
                "current": job.get("progress", {}).get("current", 0),
                "total":   4,
                "message": f"Error: {e}",
            },
        })
        print(f"Workflow error:\n{details}")
        _save_job_to_store(job)   # persist errored job (gap 5)
    finally:
        loop.close()


async def _run_workflow(research_id: str, company_data: Dict):
    job            = research_jobs[research_id]
    company_name   = company_data["company_name"]
    financial_data = company_data["financial_data"]
    gemini_key     = get_gemini_key()   # snapshot at start of job

    # Step 1 ────────────────────────────────────────────────────────────────
    job.update({
        "status":       "running",
        "current_step": "company_selected",
        "progress": {
            "step":    "Step 1: Company Selected",
            "current": 1, "total": 4,
            "message": f"Selected: {company_name}",
        },
    })
    await asyncio.sleep(0.3)

    # Step 2 — Web research (+ AI query generation) ─────────────────────────
    job["current_step"] = "research_agent"
    job["progress"] = {
        "step":    "Step 2: Web Research",
        "current": 2, "total": 4,
        "message": "Generating AI search queries…",
    }

    # Gap 1+2+3: pass gemini_key so ResearchAgent can generate queries,
    # fetch page content, and skip cached categories.
    agent = ResearchAgent(output_dir=RESEARCH_OUTPUT_DIR, gemini_api_key=gemini_key)

    def research_cb(info):
        job["progress"]["details"] = info
        phase = info.get("phase")
        if phase == "query_generation":
            job["progress"]["message"] = info.get("message", "Generating AI queries…")
        elif info.get("cached"):
            job["progress"]["message"] = info.get("message", "Loading from cache…")
        elif info.get("subtopic"):
            cat  = info.get("category", "").replace("_", " ").title()
            src  = " [AI]" if info.get("query_source") == "ai" else ""
            job["progress"]["message"] = f"[{cat}]{src} {info.get('message', '…')}"
        elif info.get("category"):
            job["progress"]["message"] = info.get("message", "Processing…")

    agent.run_research(
        company_name=company_name,
        financial_data=financial_data,
        progress_callback=research_cb,
    )

    # Step 3 — Report generation ────────────────────────────────────────────
    job["current_step"] = "report_generation"
    job["progress"] = {
        "step":    "Step 3: Generating Report",
        "current": 3, "total": 4,
        "message": "Generating analyst report with Gemini…",
    }

    summ = SummarizationAgent(gemini_api_key=gemini_key, model=_config["model"])
    research_data = summ.load_research_outputs(RESEARCH_OUTPUT_DIR, company_name)
    if not research_data:
        raise Exception("No research data found — cannot generate report")

    report = summ.create_analyst_report(company_name, financial_data, research_data)
    if not report or report.startswith("Error"):
        raise Exception(f"Report generation failed: {report}")

    # Step 4 — Validation + TLDR (one combined Gemini call) ────────────────
    job["current_step"] = "validation"
    job["progress"] = {
        "step":    "Step 4: Validation & Summary",
        "current": 4, "total": 4,
        "message": "Validating buy/avoid + generating summary (1 combined call)…",
    }

    # validate_and_summarize merges validate_buy_avoid + generate_tldr_summary
    # into a single Gemini call, saving one API request on the free tier.
    combined = summ.validate_and_summarize(company_name, financial_data, research_data, report)
    if not combined or combined.get("recommendation") == "ERROR":
        raise Exception(f"Validation failed: {combined.get('error', 'Unknown')}")

    tldr       = combined.pop("tldr", "") or ""   # extract before passing to save_report
    validation = combined                          # remaining keys match old validate_buy_avoid shape

    # Pass pre-built tldr so save_report skips the extra generate_tldr_summary call
    report_path, tldr = summ.save_report(company_name, report, validation, REPORTS_DIR, tldr=tldr)

    # Accumulate token usage into global counter
    _accumulate_tokens(summ)

    # Done ──────────────────────────────────────────────────────────────────
    job.update({
        "status":       "completed",
        "current_step": "completed",
        "progress": {
            "step":    "Completed",
            "current": 4, "total": 4,
            "message": "Research completed successfully!",
        },
        "results": {
            "report":         report,
            "tldr":           tldr,           # Gap 4: now included
            "validation":     validation,
            "report_path":    report_path,
            "recommendation": validation.get("recommendation", "N/A"),
            "confidence":     validation.get("confidence", "N/A"),
            "tokens_used":    summ.token_stats,
        },
        "completed_at": datetime.now().isoformat(),
    })

    # Gap 5: persist the completed job
    _save_job_to_store(job)


@app.get("/research/status/{research_id}", response_model=ResearchStatus)
async def get_research_status(research_id: str):
    if research_id not in research_jobs:
        raise HTTPException(status_code=404, detail="Research ID not found")
    job = research_jobs[research_id]
    return ResearchStatus(
        research_id=job["research_id"],
        status=job["status"],
        company_name=job["company_name"],
        progress=job["progress"],
        current_step=job["current_step"],
        error=job.get("error"),
    )


@app.get("/research/results/{research_id}")
async def get_research_results(research_id: str):
    if research_id not in research_jobs:
        raise HTTPException(status_code=404, detail="Research ID not found")
    job = research_jobs[research_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Not completed. Status: {job['status']}")
    return job["results"]


@app.get("/reports/list")
async def list_reports():
    try:
        reports_dir = Path(REPORTS_DIR)
        files = list(reports_dir.glob("*_Analyst_Report_*.md")) if reports_dir.exists() else []
        reports = []
        for f in files:
            parts = f.name.split("_Analyst_Report_")
            reports.append({
                "filename":     f.name,
                "company_name": parts[0].replace("_", " ") if len(parts) > 1 else f.stem,
                "date":         parts[1].replace(".md", "") if len(parts) > 1 else "",
                "size":         f.stat().st_size,
                "modified":     datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
        return {"reports": sorted(reports, key=lambda x: x["modified"], reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/{company_name}")
async def download_report(company_name: str):
    try:
        reports_dir = Path(REPORTS_DIR)
        safe  = "".join(c for c in company_name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
        files = list(reports_dir.glob(f"{safe}_Analyst_Report_*.md")) if reports_dir.exists() else []
        if not files:
            raise HTTPException(status_code=404, detail=f"No report for {company_name}")
        latest = max(files, key=lambda p: p.stat().st_mtime)
        return FileResponse(path=latest, filename=latest.name, media_type="text/markdown")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
