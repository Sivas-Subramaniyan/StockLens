"""
Evidence-Gathering Research Agent
Performs web searches using DuckDuckGo (free) and accumulates evidence across 12 predefined categories.

Improvements:
  - Gap 1: Gemini generates optimised search queries (one API call, all categories at once)
  - Gap 2: Top-2 URLs per query are fetched for full page content (not just snippets)
  - Gap 3: Per-category caching — already-searched categories are skipped on re-run
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from pathlib import Path

import requests as _requests

# Fix Windows encoding issues
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


class ResearchAgent:
    """
    Evidence-gathering research agent that searches the web using DuckDuckGo
    and accumulates factual evidence about companies mapped to predefined business categories.

    New in this version:
      - Gemini-generated search queries (gap 1)
      - Full page content fetching for top 2 results per query (gap 2)
      - Per-category caching — skip categories already researched today (gap 3)
    """

    def __init__(self, output_dir: str = "research_output", gemini_api_key: str = ""):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.gemini_api_key = gemini_api_key
        self.company_name = ""
        self.financial_data = {}
        self.retrieval_date = datetime.now().strftime("%Y-%m-%d")
        self.search_categories = self._define_search_categories()

    # ── Search category definitions ────────────────────────────────────────

    def _define_search_categories(self) -> Dict[str, List[str]]:
        """Define all 12 search categories with their subtopics."""
        return {
            "1_business_fundamentals_and_model_stability": [
                "core business description and primary value proposition",
                "segment-wise revenue and profit breakdown",
                "revenue concentration and stability of key contracts",
                "competitive landscape and differentiation factors",
            ],
            "2_financial_strength_and_quality_of_earnings": [
                "5-year trend revenue EBITDA operating profit PAT",
                "cash flow consistency CFO vs PAT FCF sustainability",
                "ROE ROCE ROA trend vs industry averages",
                "margin stability gross operating net margins",
            ],
            "3_balance_sheet_health_and_liquidity": [
                "debt-to-equity ratio interest coverage leverage trend",
                "cash and liquid assets vs short-term obligations",
                "capital expenditure trend maintenance vs growth capex",
            ],
            "4_intrinsic_value_and_market_positioning": [
                "current market price market cap enterprise value valuation",
                "analyst target price consensus valuation estimates",
                "DCF comparable P/E EV/EBITDA P/B fair value estimation",
            ],
            "5_economic_moat_and_durability": [
                "sources of moat brand equity IP patents regulatory licenses switching costs",
                "evidence of pricing power gross margin resilience market share stability",
                "network effects ecosystem lock-ins customer loyalty renewal rates",
            ],
            "6_management_integrity_and_capital_allocation": [
                "key management bios track record tenure competence",
                "insider ownership recent insider trading buy sell trends",
                "capital allocation track record acquisitions buybacks dividends",
                "governance indicators board independence audit quality disclosures",
            ],
            "7_growth_drivers_and_future_visibility": [
                "strategic initiatives expansion plans R&D product pipeline partnerships",
                "industry growth projections and tailwinds",
                "company growth guidance vs historical delivery rate",
            ],
            "8_macro_and_regional_sensitivity": [
                "dependence on domestic vs export markets FX sensitivity",
                "regulatory dependencies policy changes taxation impact",
            ],
            "9_behavioral_and_market_sentiment": [
                "12-month major news litigation fraud leadership change contracts",
                "analyst rating distribution and changes FII DII flow trends",
            ],
            "10_risks_and_downside_scenarios": [
                "structural industry risks technology obsolescence policy threats",
                "execution risks management capability delays capex product rollout",
                "financial risks leverage liquidity crunch credit events",
                "governance compliance risks audit issues insider conflicts",
            ],
            "11_integrity_and_governance_health": [
                "related-party transactions promoter pledging trends",
                "corporate governance ratings regulatory penalties SEBI actions",
                "litigation record material legal exposures ESG controversies",
            ],
            "12_overall_fundamental_conviction": [
                "stability across cycles earnings resilience past downturns",
                "cash flow predictability margin durability management credibility",
                "valuation comfort vs fundamentals net upside-to-risk trade-off",
            ],
        }

    # ── Gap 1: AI query generation ─────────────────────────────────────────

    def _generate_all_queries(self, progress_callback: Optional[callable] = None) -> Dict[str, Dict[str, str]]:
        """
        One Gemini call that returns an optimised search query for every
        category/subtopic pair.  Falls back to an empty dict on any error
        so the caller can continue with default queries.
        """
        if not self.gemini_api_key:
            return {}

        # Skip expensive Gemini call when every category is already cached today
        if self._all_categories_cached():
            print("  [AI Queries] All categories cached — skipping query generation")
            if progress_callback:
                progress_callback({
                    "phase": "query_generation",
                    "message": "All categories cached — skipping query generation",
                })
            return {}

        if progress_callback:
            progress_callback({
                "phase": "query_generation",
                "message": "Gemini generating optimised search queries…",
            })

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.gemini_api_key)

            fd_lines = "\n".join(
                f"  - {k}: {v}"
                for k, v in self.financial_data.items()
                if v and str(v) not in ("nan", "")
            )
            cats_json = {cat: subtopics for cat, subtopics in self.search_categories.items()}

            prompt = f"""You are a financial research specialist for Indian smallcap stocks.

Company: {self.company_name}
Key financials:
{fd_lines or '  (none provided)'}

For EVERY category and subtopic listed below generate ONE highly targeted search query
that will surface the most useful financial news, analyst data, or company disclosures.
Tailor each query to the specific company name AND the subtopic intent.

Categories:
{json.dumps(cats_json, indent=2)}

Return ONLY valid JSON — no markdown fences, no extra text — in exactly this shape:
{{
  "<category_name>": {{
    "<subtopic>": "<search query>"
  }}
}}"""

            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=4096,
                temperature=0.1,
            )

            resp = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt,
                config=config,
            )

            text = resp.text
            if text is None and resp.candidates:
                parts = resp.candidates[0].content.parts or []
                text = "".join(p.text for p in parts if getattr(p, "text", None)) or None

            if text:
                parsed = json.loads(text)
                total_queries = sum(len(v) for v in parsed.values())
                print(f"  [AI Queries] Generated {total_queries} optimised queries across {len(parsed)} categories")
                if progress_callback:
                    progress_callback({
                        "phase": "query_generation",
                        "message": f"AI generated {total_queries} targeted queries — starting web research…",
                    })
                return parsed

        except Exception as e:
            print(f"  [AI Queries] Generation failed ({e}), using default queries")
            if progress_callback:
                progress_callback({
                    "phase": "query_generation",
                    "message": "Query generation unavailable — using default queries",
                })

        return {}

    # ── Gap 2: Full page content fetching ──────────────────────────────────

    def _fetch_page_content(self, url: str, timeout: int = 5) -> str:
        """
        Fetch a URL and return clean plain text (up to 600 words).
        Returns empty string on any error — never raises.
        """
        if not url or not url.startswith("http"):
            return ""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp = _requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            if resp.status_code != 200:
                return ""
            html = resp.text
            # Strip <script> and <style> blocks
            html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html,
                          flags=re.DOTALL | re.IGNORECASE)
            # Strip all remaining tags
            text = re.sub(r"<[^>]+>", " ", html)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            # Return first 600 words
            words = text.split()
            return " ".join(words[:600])
        except Exception:
            return ""

    # ── Gap 3: Per-category caching ────────────────────────────────────────

    def _cache_filepath(self, category_name: str) -> Path:
        safe = (
            "".join(c for c in self.company_name if c.isalnum() or c in (" ", "-", "_"))
            .strip()
            .replace(" ", "_")
        )
        return self.output_dir / f"{category_name}_{safe}_{self.retrieval_date}.json"

    def _is_cached(self, category_name: str) -> bool:
        """Return True if this category has already been searched today."""
        return self._cache_filepath(category_name).exists()

    def _all_categories_cached(self) -> bool:
        """Return True if every category is already cached for today's date."""
        return all(self._is_cached(cat) for cat in self.search_categories)

    # ── Core search helpers ────────────────────────────────────────────────

    def search_web(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        DuckDuckGo search (10 results by default) + parallel full-page content
        fetch for the top 2 URLs.

        URL fetches run concurrently (ThreadPoolExecutor, max 2 workers) so the
        per-query overhead is ~3-5 s instead of ~8-10 s for sequential fetches.
        """
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results, timelimit="y"))

            # Parallel-fetch full content for top 2 URLs
            urls_to_fetch = [
                r.get("href", "") for r in raw[:2]
                if r.get("href", "").startswith("http")
            ]
            fetched: Dict[str, str] = {}
            if urls_to_fetch:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    future_to_url = {pool.submit(self._fetch_page_content, u): u
                                     for u in urls_to_fetch}
                    for fut in as_completed(future_to_url):
                        url = future_to_url[fut]
                        try:
                            fetched[url] = fut.result()
                        except Exception:
                            fetched[url] = ""

            results = []
            for r in raw:
                url  = r.get("href", "")
                body = r.get("body", "")
                full = fetched.get(url, "")
                results.append({
                    "url":            url,
                    "title":          r.get("title", ""),
                    "source_domain":  self._extract_domain(url),
                    "retrieval_date": self.retrieval_date,
                    "excerpt":        self._truncate_excerpt(body, max_words=100),
                    "full_content":   full[:1200] if full else "",
                    "confidence":     self._assess_confidence(url),
                    "raw_content":    body[:2000],
                })
            return results
        except Exception as e:
            print(f"  [Search error] {e}")
            return []

    def _extract_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc or ""
        except Exception:
            return ""

    def _truncate_excerpt(self, text: str, max_words: int = 100) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "..."

    def _assess_confidence(self, url: str) -> str:
        url_lower = url.lower()
        domain = self._extract_domain(url_lower)
        high   = ["sec.gov", "sebi.gov.in", "mca.gov.in", "bloomberg.com", "reuters.com",
                  "factset.com", "finance.yahoo.com"]
        medium = ["seekingalpha.com", "morningstar.com", "financialexpress.com",
                  "economictimes.indiatimes.com", "livemint.com", "moneycontrol.com"]
        for d in high:
            if d in domain or d in url_lower:
                return "high"
        for d in medium:
            if d in domain or d in url_lower:
                return "medium"
        return "medium"

    # ── Category search ────────────────────────────────────────────────────

    def search_category(
        self,
        category_name: str,
        subtopics: List[str],
        progress_callback: Optional[callable] = None,
        ai_queries: Optional[Dict[str, str]] = None,   # {subtopic: query}
    ) -> Dict[str, Any]:
        """Search all subtopics for a category and accumulate results."""
        print(f"\n{'='*50}")
        print(f"Category: {category_name}")
        print(f"{'='*50}")

        category_results = {
            "category":               category_name,
            "company_name":           self.company_name,
            "retrieval_date":         self.retrieval_date,
            "financial_data_provided": self.financial_data,
            "subtopics":              {},
        }

        for i, subtopic in enumerate(subtopics, 1):
            # Pick the best available query (AI-generated > default)
            if ai_queries and subtopic in ai_queries:
                query = ai_queries[subtopic]
                query_source = "ai"
            else:
                query = f"{self.company_name} India {subtopic}"
                query_source = "default"

            print(f"  [{i}/{len(subtopics)}] {subtopic}")
            print(f"    Query ({query_source}): {query}")

            if progress_callback:
                progress_callback({
                    "category":        category_name,
                    "subtopic":        subtopic,
                    "subtopic_number": i,
                    "total_subtopics": len(subtopics),
                    "query_source":    query_source,
                    "message":         f"Searching: {subtopic}",
                })

            try:
                results = self.search_web(query, max_results=8)
                has_content = sum(1 for r in results if r.get("full_content"))
                category_results["subtopics"][subtopic] = {
                    "query":          query,
                    "query_source":   query_source,
                    "results_count":  len(results),
                    "pages_fetched":  has_content,
                    "evidence":       results,
                }
                print(f"    Found {len(results)} results, {has_content} pages fetched")
                if progress_callback:
                    progress_callback({
                        "category":        category_name,
                        "subtopic":        subtopic,
                        "subtopic_number": i,
                        "total_subtopics": len(subtopics),
                        "results_found":   len(results),
                        "pages_fetched":   has_content,
                        "message":         f"Found {len(results)} results, {has_content} pages fetched",
                    })
            except Exception as e:
                print(f"    Error: {e}")
                category_results["subtopics"][subtopic] = {
                    "query":         query,
                    "query_source":  query_source,
                    "results_count": 0,
                    "pages_fetched": 0,
                    "evidence":      [],
                    "error":         str(e),
                }

            # Polite delay between DuckDuckGo searches (1 s is enough; was 1.5 s)
            time.sleep(1.0)

        return category_results

    def save_category_results(self, category_name: str, results: Dict[str, Any]) -> str:
        filepath = self._cache_filepath(category_name)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {filepath}")
        return str(filepath)

    # ── Main research runner ───────────────────────────────────────────────

    def run_research(
        self,
        company_name: str,
        financial_data: Optional[Dict] = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, str]:
        """Run complete research across all categories."""
        self.company_name   = company_name
        self.financial_data = financial_data or {}

        print(f"\n{'#'*50}")
        print(f"Starting Research: {company_name}")
        print(f"Date: {self.retrieval_date}")
        print(f"{'#'*50}\n")

        # ── Gap 1: Generate AI queries once for all categories ────────────
        ai_queries_all = self._generate_all_queries(progress_callback)

        saved_files    = {}
        total_categories = len(self.search_categories)
        cached_count   = 0

        for cat_idx, (category_name, subtopics) in enumerate(self.search_categories.items(), 1):

            # ── Gap 3: Skip if already cached today ───────────────────────
            if self._is_cached(category_name):
                cached_count += 1
                filepath = str(self._cache_filepath(category_name))
                saved_files[category_name] = filepath
                print(f"\n  [Cache] {category_name} — skipping (already researched today)")
                if progress_callback:
                    progress_callback({
                        "category":         category_name,
                        "category_number":  cat_idx,
                        "total_categories": total_categories,
                        "cached":           True,
                        "subtopic":         None,
                        "message":          f"[Cache] {cat_idx}/{total_categories}: {category_name.replace('_', ' ').title()}",
                    })
                continue

            # ── Run search for this category ──────────────────────────────
            try:
                if progress_callback:
                    progress_callback({
                        "category":         category_name,
                        "category_number":  cat_idx,
                        "total_categories": total_categories,
                        "cached":           False,
                        "subtopic":         None,
                        "message":          f"Processing {cat_idx}/{total_categories}: {category_name.replace('_', ' ').title()}",
                    })

                # Pass the AI-generated queries for this category (if any)
                cat_ai_queries = ai_queries_all.get(category_name, {})

                results  = self.search_category(category_name, subtopics, progress_callback, cat_ai_queries)
                filepath = self.save_category_results(category_name, results)
                saved_files[category_name] = filepath

            except Exception as e:
                print(f"\n[ERROR] {category_name}: {e}")
                if progress_callback:
                    progress_callback({
                        "category": category_name,
                        "error":    str(e),
                        "message":  f"Error in {category_name}: {str(e)}",
                    })
                saved_files[category_name] = None

        # ── Save summary ───────────────────────────────────────────────────
        safe = (
            "".join(c for c in self.company_name if c.isalnum() or c in (" ", "-", "_"))
            .strip()
            .replace(" ", "_")
        )
        summary = {
            "company_name":        self.company_name,
            "research_date":       self.retrieval_date,
            "financial_data":      self.financial_data,
            "ai_queries_used":     bool(ai_queries_all),
            "categories_completed": len([f for f in saved_files.values() if f]),
            "categories_cached":   cached_count,
            "category_files":      saved_files,
        }
        summary_path = self.output_dir / f"summary_{safe}_{self.retrieval_date}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'#'*50}")
        print(f"Research Complete! {cached_count} categories from cache.")
        print(f"Summary: {summary_path}")
        print(f"{'#'*50}\n")
        return saved_files
