"""
Summarization and Validation Agent
Uses Google Gemini 2.5 Flash (free tier) to generate analyst reports and validate buy/avoid decisions.
Optimised for free tier: 10 RPM. Tracks token usage per session.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')


class SummarizationAgent:
    """
    Generates comprehensive analyst reports and validates buy/avoid decisions
    using Google Gemini (free tier, new google-genai SDK).
    Tracks token usage per session.
    """

    DEFAULT_MODEL = "models/gemini-2.5-flash"

    # Fallback model order if primary is unavailable
    FALLBACK_MODELS = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-flash-latest",
    ]

    def __init__(self, gemini_api_key: str, model: str = DEFAULT_MODEL):
        from google import genai
        self._client = genai.Client(api_key=gemini_api_key)
        self.model = model
        self._last_call_time = 0.0

        # Token usage tracking for this session
        self._token_stats: Dict = {
            "input_tokens":  0,
            "output_tokens": 0,
            "total_tokens":  0,
            "api_calls":     0,
            "model_used":    model,
        }

    @property
    def token_stats(self) -> Dict:
        return dict(self._token_stats)

    # ── Rate limiting: free tier safe (10 RPM → 7 s min between calls) ──
    def _rate_limit(self, min_interval: float = 7.0):
        elapsed = time.time() - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()

    def _record_usage(self, response, model: str):
        """Extract token counts from Gemini response metadata."""
        try:
            um = response.usage_metadata
            if um:
                inp = getattr(um, 'prompt_token_count', 0) or 0
                out = getattr(um, 'candidates_token_count', 0) or 0
                # 2.5-flash also has thoughts_token_count
                thoughts = getattr(um, 'thoughts_token_count', 0) or 0
                total = getattr(um, 'total_token_count', 0) or (inp + out + thoughts)

                self._token_stats["input_tokens"]  += inp
                self._token_stats["output_tokens"] += out + thoughts
                self._token_stats["total_tokens"]  += total
                self._token_stats["api_calls"]     += 1
                self._token_stats["model_used"]     = model
        except Exception:
            # Don't let token tracking break the main flow
            self._token_stats["api_calls"] += 1

    def _generate(self, prompt: str, system: str, json_mode: bool,
                  temperature: float, max_tokens: int) -> str:
        """
        Core generation call with retry + model fallback.
        Retries up to 3 times per model, then tries fallback models on 503.
        """
        from google.genai import types

        models_to_try = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]

        for model in models_to_try:
            config_kwargs = dict(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system if system else None,
            )
            if json_mode:
                config_kwargs["response_mime_type"] = "application/json"
            config = types.GenerateContentConfig(**config_kwargs)

            for attempt in range(1, 4):  # up to 3 attempts per model
                self._rate_limit()
                try:
                    response = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    if model != self.model:
                        print(f"  [Gemini] Using fallback model: {model}")

                    # Record token usage BEFORE potentially raising
                    self._record_usage(response, model)

                    # gemini-2.5-flash uses thinking tokens; response.text may be None
                    # when token budget is exhausted — extract from parts as fallback
                    text = response.text
                    if text is None and response.candidates:
                        parts = response.candidates[0].content.parts or []
                        text = "".join(
                            p.text for p in parts if getattr(p, "text", None)
                        ) or None
                    if text is not None:
                        return text
                    raise ValueError("Model returned empty response (max_output_tokens may be too low)")

                except Exception as e:
                    err_str = str(e)
                    is_503 = "503" in err_str or "UNAVAILABLE" in err_str
                    is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                    if is_503:
                        wait = 15 * attempt  # 15 s → 30 s → 45 s
                        print(f"  [Gemini] 503 on {model} attempt {attempt}/3 — waiting {wait}s…")
                        time.sleep(wait)
                    elif is_429:
                        wait = 20 * attempt
                        print(f"  [Gemini] 429 rate-limit on {model} — waiting {wait}s…")
                        time.sleep(wait)
                    else:
                        print(f"  [Gemini] Error on {model}: {err_str[:120]}")
                        break  # non-retriable — try next model

            print(f"  [Gemini] All attempts failed for {model}, trying next…")

        raise RuntimeError("All Gemini models exhausted. Please retry in a few minutes.")

    def _call_text(self, prompt: str, system: str = "", temperature: float = 0.2,
                   max_tokens: int = 8192) -> str:
        return self._generate(prompt, system, json_mode=False,
                               temperature=temperature, max_tokens=max_tokens)

    def _call_json(self, prompt: str, system: str = "", temperature: float = 0.1,
                   max_tokens: int = 4096) -> dict:
        raw = self._generate(prompt, system, json_mode=True,
                              temperature=temperature, max_tokens=max_tokens)
        return json.loads(raw)

    # ── Load research outputs ────────────────────────────────────────────
    def load_research_outputs(self, research_output_dir: str, company_name: str) -> Dict:
        research_dir = Path(research_output_dir)
        safe = "".join(
            c for c in company_name if c.isalnum() or c in (' ', '-', '_')
        ).strip().replace(' ', '_')

        research_data = {}
        for f in research_dir.glob(f"*_{safe}_*.json"):
            if f.name.startswith("summary_"):
                continue
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                    research_data[data.get('category', f.stem)] = data
            except Exception as e:
                print(f"Error loading {f}: {e}")
        return research_data

    # ── Build research text ──────────────────────────────────────────────
    def _research_to_text(self, research_data: Dict, max_chars: int = 60_000,
                          max_items: int = 5, excerpt_chars: int = 150,
                          full_content_chars: int = 250) -> str:
        """
        Convert research JSON into a compact text block for Gemini.

        Defaults (tuned for free-tier token budget):
          max_chars=60_000   — hard cap on total output size
          max_items=5        — top evidence items per subtopic (was 6)
          excerpt_chars=150  — DuckDuckGo snippet length  (was 200)
          full_content_chars=250 — fetched page content length (was 400)
        """
        parts = []
        for cat_name, cat_data in research_data.items():
            parts.append(f"\n{'='*60}")
            parts.append(cat_name.replace('_', ' ').upper())
            parts.append('='*60)
            for subtopic, sub_data in cat_data.get('subtopics', {}).items():
                q_source = sub_data.get('query_source', 'default')
                pages    = sub_data.get('pages_fetched', 0)
                parts.append(f"\nSubtopic: {subtopic}  [query:{q_source}  pages:{pages}]")
                for i, item in enumerate(sub_data.get('evidence', [])[:max_items], 1):
                    parts.append(f"\n  {i}. {item.get('title', 'N/A')}")
                    parts.append(f"     Source: {item.get('source_domain', '')}")
                    excerpt = item.get('excerpt', '')[:excerpt_chars]
                    if excerpt:
                        parts.append(f"     Snippet: {excerpt}")
                    # Only include fetched page content when it adds meaningful text
                    full = item.get('full_content', '').strip()
                    if len(full) > 80:
                        parts.append(f"     Content: {full[:full_content_chars]}")
        text = "\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n… [truncated for token budget]"
        return text

    def _financial_to_text(self, financial_data: Dict) -> str:
        return "\n".join(
            f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in financial_data.items()
            if v and str(v) not in ('nan', '')
        )

    # ── Create analyst report (1 Gemini call) ───────────────────────────
    def create_analyst_report(self, company_name: str, financial_data: Dict,
                               research_data: Dict) -> str:
        research_text  = self._research_to_text(research_data)
        financial_text = self._financial_to_text(financial_data)

        system = (
            "You are a STRICT and CRITICAL equity research analyst specialising in Indian smallcap stocks. "
            "Default to AVOID unless there is OVERWHELMING evidence for BUY. "
            "Prioritise risk management, fraud detection, and governance over potential upside."
        )

        prompt = f"""Create a comprehensive analyst report for **{company_name}**.

## Financial Data
{financial_text}

## Research Evidence (from web search)
{research_text}

## Instructions
1. Open with a bold **BUY** or **AVOID** recommendation.
2. Default to AVOID unless evidence strongly supports BUY.
3. Prioritise risk analysis first: fraud, SEBI actions, governance, accounting issues, promoter pledging.
4. Cover each research category systematically.
5. Assess 40%+ return probability in 3 years — be conservative.
6. Use markdown with clear headings. Cite source domains where possible.
7. Weaknesses and risks FIRST, then strengths."""

        try:
            return self._call_text(prompt, system=system, temperature=0.2, max_tokens=8192)
        except Exception as e:
            return f"Error generating report: {e}"

    # ── Validate buy/avoid (1 Gemini call, JSON) ────────────────────────
    def validate_buy_avoid(self, company_name: str, financial_data: Dict,
                            research_data: Dict, report: str) -> Dict:
        research_text  = self._research_to_text(research_data, max_chars=20_000,
                                                max_items=3, excerpt_chars=100,
                                                full_content_chars=150)
        report_snippet = report[:6000] + "…[truncated]" if len(report) > 6000 else report
        fd = financial_data

        system = (
            "You are a STRICT quantitative analyst for Indian equities. "
            "Default to AVOID when uncertain. Prioritise risk over upside."
        )

        prompt = f"""Evaluate **{company_name}** for a BUY / AVOID decision.

## Key Financials
- Market Cap: {fd.get('market_cap', 'N/A')} Cr  |  P/E: {fd.get('pe_ratio', 'N/A')}
- ROCE: {fd.get('roce', 'N/A')}%  |  CMP: {fd.get('CMP Rs.', 'N/A')}
- Promoter Holding: {fd.get('prom_hold', 'N/A')}%
- FII Change: {fd.get('chg_fii', 'N/A')}%  |  DII Change: {fd.get('chg_dii', 'N/A')}%
- Debt/Equity: {fd.get('debt_eq', 'N/A')}  |  FCF 3Y: {fd.get('fcf_3y', 'N/A')} Cr
- Working Capital Days: {fd.get('wc_days', 'N/A')}  |  Cash Cycle: {fd.get('cash_cycle', 'N/A')}
- Industry P/E: {fd.get('ind_pe', 'N/A')}  |  Investment Score: {fd.get('investment_score', 'N/A')}

## Research Summary
{research_text}

## Analyst Report
{report_snippet}

## Rules
- BUY: strong financials, no material red flags, clear moat, 40%+ return highly probable.
- AVOID: any material risk, fraud, weak financials, uncertain upside. When in doubt → AVOID.

Return JSON with exactly these keys:
{{
  "recommendation": "BUY" or "AVOID",
  "confidence": "high" or "medium" or "low",
  "expected_return_3y": "e.g. 45% or N/A",
  "probability_40pct_return": "high or medium or low",
  "key_drivers": ["driver1"],
  "key_risks": ["risk1"],
  "red_flags_found": ["flag1"],
  "financial_concerns": ["concern1"],
  "reasoning": "2-3 sentence explanation"
}}"""

        try:
            return self._call_json(prompt, system=system, temperature=0.1, max_tokens=2048)
        except Exception as e:
            return {"recommendation": "ERROR", "error": str(e), "reasoning": "Validation failed"}

    # ── TLDR summary — standalone fallback (1 extra Gemini call) ───────────
    def generate_tldr_summary(self, company_name: str, validation: Dict, report: str) -> str:
        """Kept for backward-compat (e.g. research_orchestrator). Prefer validate_and_summarize."""
        rec       = validation.get('recommendation', 'N/A')
        conf      = validation.get('confidence', 'N/A')
        ret       = validation.get('expected_return_3y', 'N/A')
        reasoning = validation.get('reasoning', '')[:400]

        prompt = f"""Write a 3-5 sentence professional executive summary paragraph for:

Company: {company_name}  |  Recommendation: {rec}  |  Confidence: {conf}
Expected 3Y Return: {ret}  |  Key Reasoning: {reasoning}

Requirements: state recommendation upfront, give key rationale, mention expected return and confidence.
Output ONLY the paragraph — no headings, no markdown, no extra text."""

        try:
            return self._call_text(prompt, temperature=0.3, max_tokens=512).strip()
        except Exception as e:
            return f"{company_name}: {rec} recommendation ({conf} confidence). Expected 3Y return: {ret}. {reasoning}"

    # ── Combined validate + TLDR (saves 1 Gemini call vs separate calls) ──
    def validate_and_summarize(self, company_name: str, financial_data: Dict,
                               research_data: Dict, report: str) -> Dict:
        """
        Single Gemini call that returns both the buy/avoid validation AND
        a 3-5 sentence TLDR executive summary.

        Replaces the separate validate_buy_avoid() + generate_tldr_summary() pair,
        saving one API call per research session on the free tier.

        Returns the same keys as validate_buy_avoid() plus a "tldr" key.
        Falls back to separate calls if the combined call fails.
        """
        research_text  = self._research_to_text(research_data, max_chars=25_000)
        report_snippet = report[:6000] + "…[truncated]" if len(report) > 6000 else report
        fd = financial_data

        system = (
            "You are a STRICT quantitative analyst for Indian equities. "
            "Default to AVOID when uncertain. Prioritise capital preservation over upside."
        )

        prompt = f"""Evaluate **{company_name}** for a BUY / AVOID decision AND write a concise executive summary.

## Key Financials
- Market Cap: {fd.get('market_cap', 'N/A')} Cr  |  P/E: {fd.get('pe_ratio', 'N/A')}
- ROCE: {fd.get('roce', 'N/A')}%  |  CMP: {fd.get('CMP Rs.', 'N/A')}
- Promoter Holding: {fd.get('prom_hold', 'N/A')}%
- FII Change: {fd.get('chg_fii', 'N/A')}%  |  DII Change: {fd.get('chg_dii', 'N/A')}%
- Debt/Equity: {fd.get('debt_eq', 'N/A')}  |  FCF 3Y: {fd.get('fcf_3y', 'N/A')} Cr
- Working Capital Days: {fd.get('wc_days', 'N/A')}  |  Cash Cycle: {fd.get('cash_cycle', 'N/A')}
- Industry P/E: {fd.get('ind_pe', 'N/A')}  |  Investment Score: {fd.get('investment_score', 'N/A')}

## Research Evidence
{research_text}

## Analyst Report
{report_snippet}

## Decision Rules
- BUY only if: strong financials, no material red flags, clear moat, 40%+ return highly probable.
- AVOID if: any material risk, fraud, governance issues, uncertain upside. When in doubt → AVOID.

Return ONLY valid JSON with exactly these keys:
{{
  "recommendation": "BUY" or "AVOID",
  "confidence": "high" or "medium" or "low",
  "expected_return_3y": "e.g. 45% or N/A",
  "probability_40pct_return": "high or medium or low",
  "key_drivers": ["driver1", "driver2"],
  "key_risks": ["risk1", "risk2"],
  "red_flags_found": ["flag1"],
  "financial_concerns": ["concern1"],
  "reasoning": "2-3 sentence explanation of the decision",
  "tldr": "3-5 sentence professional executive summary. State the recommendation upfront, give the key rationale, mention expected return and confidence level."
}}"""

        try:
            result = self._call_json(prompt, system=system, temperature=0.1, max_tokens=2560)
            # Ensure tldr is present; fall back to reasoning if missing
            if not result.get("tldr"):
                result["tldr"] = result.get("reasoning", "")
            return result
        except Exception as e:
            print(f"  [Gemini] validate_and_summarize failed ({e}), falling back to separate calls")
            # Graceful fallback: run the two original calls independently
            validation = self.validate_buy_avoid(company_name, financial_data, research_data, report)
            validation["tldr"] = self.generate_tldr_summary(company_name, validation, report)
            return validation

    # ── Save report ──────────────────────────────────────────────────────
    def save_report(self, company_name: str, report: str, validation: Dict,
                    output_dir: str = "reports", tldr: str = None) -> tuple:
        """
        Saves the full analyst report to disk.

        Args:
            tldr: Pre-generated TLDR string (from validate_and_summarize).
                  If None, generates it with an extra Gemini call (backward-compat).

        Returns:
            (filepath: str, tldr: str)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        safe = "".join(
            c for c in company_name if c.isalnum() or c in (' ', '-', '_')
        ).strip().replace(' ', '_')
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = output_path / f"{safe}_Analyst_Report_{date_str}.md"

        if tldr is None:
            print("\nGenerating TLDR summary (extra call — use validate_and_summarize to avoid this)…")
            tldr = self.generate_tldr_summary(company_name, validation, report)

        drivers  = "\n".join(f"- {d}" for d in validation.get('key_drivers', []))        or "- None identified"
        risks    = "\n".join(f"- {r}" for r in validation.get('key_risks', []))           or "- None identified"
        flags    = "\n".join(f"- ⚠️ {f}" for f in validation.get('red_flags_found', [])) or "- No major red flags"
        concerns = "\n".join(f"- ⚠️ {c}" for c in validation.get('financial_concerns', [])) or "- None identified"

        content = f"""# Analyst Report: {company_name}

**Date:** {date_str}

---

## Executive Summary

| | |
|---|---|
| **Recommendation** | **{validation.get('recommendation', 'N/A')}** |
| **Confidence** | {validation.get('confidence', 'N/A')} |
| **Expected 3-Year Return** | {validation.get('expected_return_3y', 'N/A')} |
| **Probability of 40%+ Return** | {validation.get('probability_40pct_return', 'N/A')} |

### TLDR

{tldr}

---

## Detailed Analysis

{report}

---

## Validation & Recommendation

### Key Drivers
{drivers}

### Key Risks
{risks}

### Red Flags
{flags}

### Financial Concerns
{concerns}

### Reasoning
{validation.get('reasoning', 'N/A')}

---

*Generated by StockLens AI (Gemini {self.model}) — {date_str}*
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Report saved: {filepath}")
        return str(filepath), tldr
