"""
Fetch Japanese stock data from Yahoo Finance and write data.json.

This script is executed by GitHub Actions on a schedule (장중 30분마다, see
.github/workflows/update-data.yml) but also runs locally:

    python scripts/fetch_data.py                     # writes ./data.json
    python scripts/fetch_data.py --sleep 0.1         # faster, risk rate-limit
    python scripts/fetch_data.py --limit 10          # first 10 tickers only

Exit codes:
    0   success (all or most tickers fetched)
    1   too many failures (>20% missing)           — Actions will fail loudly
    2   config or IO error
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


# ============================================================================
# Field-level conversions
# ============================================================================
def clean_nans(obj):
    """Recursively replace NaN / Infinity with None so the result is valid JSON."""
    import math
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# yfinance returns values in mixed units (some percent, some decimal). These
# helpers normalize everything to match the UI's expectations.

def to_pct(v: Any, default: float | None = None) -> float | None:
    """Convert a decimal (0.146) to percent (14.6). Rounds to 2 dp."""
    if v is None or pd.isna(v):
        return default
    return round(float(v) * 100, 2)


def to_ratio(v: Any, default: float | None = None) -> float | None:
    """Round a plain ratio, keep units."""
    if v is None or pd.isna(v):
        return default
    return round(float(v), 2)


def safe(v: Any, default: Any = None) -> Any:
    """Return v unless it is None / NaN."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return default
    return v

def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities from RSS title strings."""
    import re, html
    text = re.sub(r"<[^>]+>", "", text or "")
    return html.unescape(text).strip()

def generate_market_summary(headlines: list[dict]) -> dict | None:
    """
    Use Gemini to write a short Korean+English market tone summary
    based on the day's headlines. Returns None on failure (best-effort).
    """
    import os
    import json as _json
    import re as _re
    import urllib.request

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not headlines:
        return None

    titles = "\n".join(f"- {h['title']}" for h in headlines[:7])
    prompt = (
        "Below are today's headlines about the Japanese stock market.\n"
        "Write a 3-4 sentence market briefing in BOTH Korean and English. "
        "Include: (1) overall tone, (2) the 1-2 specific themes driving today's headlines, "
        "(3) any notable risks or catalysts mentioned. "
        "Be neutral and factual. Do not invent specifics not in the headlines. "
        "Write the Korean version in natural Korean financial-news style, not literal translation from English.\n\n"
        "Return ONLY a JSON object in this exact format, no other text, no markdown fences:\n"
        '{"ko": "Korean summary here", "en": "English summary here"}\n\n'
        f"Headlines:\n{titles}"
    )

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )
    body = _json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = _json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", errors="ignore")[:500]
        print(f"    ! gemini call failed: HTTP {e.code} {body_txt}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"    ! gemini call failed: {e}", file=sys.stderr)
        return None

    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
        # Strip optional ```json fences
        text = _re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=_re.MULTILINE)
        parsed = _json.loads(text)
        if isinstance(parsed, dict) and "ko" in parsed and "en" in parsed:
            return {
                "ko": str(parsed["ko"]).strip(),
                "en": str(parsed["en"]).strip(),
                "model": "gemini-2.5-flash",
                "generatedTs": int(time.time()),
            }
    except Exception as e:
        print(f"    ! gemini parse failed: {e}", file=sys.stderr)

    return None

def fetch_market_headlines(max_total: int = 7, max_age_hours: int = 72) -> list[dict]:
    """
    Pull Japanese-market business headlines from Google News RSS.
    Returns at most `max_total` items, fresher than `max_age_hours`.
    Best-effort — failures return an empty list, never raise.
    """
    import urllib.request
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    import time

    # Google News query — broad enough to surface major moves, narrow enough
    # to stay TSE-relevant.
    url = (
        "https://news.google.com/rss/search?"
        "q=Japan+stocks+OR+Nikkei+OR+TOPIX+OR+%22Bank+of+Japan%22+OR+yen"
        "&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            xml_data = r.read()
    except Exception as e:
        print(f"    ! headlines fetch failed: {e}", file=sys.stderr)
        return []

    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        print(f"    ! headlines parse failed: {e}", file=sys.stderr)
        return []

    cutoff = time.time() - max_age_hours * 3600
    items = []
    for item in root.iterfind(".//item"):
        title = _strip_html(item.findtext("title") or "")
        link = (item.findtext("link") or "").strip()
        pub_str = item.findtext("pubDate")
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""

        if not title or not pub_str:
            continue

        try:
            ts = parsedate_to_datetime(pub_str).timestamp()
        except Exception:
            continue

        if ts < cutoff:
            continue

        # Google News titles often end with " - SourceName"; trim if redundant.
        if source and title.endswith(f" - {source}"):
            title = title[: -(len(source) + 3)].strip()

        items.append({
            "title": title,
            "publisher": source,
            "link": link,
            "publishedTs": int(ts),
        })

    items.sort(key=lambda x: x["publishedTs"], reverse=True)
    return items[:max_total]

def compute_div_yield(info: dict) -> float | None:
    """
    Compute dividend yield from dividendRate / price — version-independent.
    yfinance's own `dividendYield` field has flipped between decimal and percent
    across versions, so we derive it ourselves.
    """
    rate = info.get("dividendRate")
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    if rate is not None and not pd.isna(rate) and price:
        return round(float(rate) / float(price) * 100, 2)

    trailing = info.get("trailingAnnualDividendYield")
    if trailing is not None and not pd.isna(trailing):
        return round(float(trailing) * 100, 2)

    return None



# ============================================================================
# Stock fetch
# ============================================================================


def fetch_stock(meta: dict[str, str], retries: int = 1) -> dict | None:
    """
    Fetch one TSE-listed stock. `meta` carries the master-list info
    (ticker, nameEn, nameKo, nameJp, sector, segment) that we trust over
    whatever Yahoo reports.
    """
    ticker_code = meta["ticker"]

    for attempt in range(retries + 1):
        try:
            yft = yf.Ticker(f"{ticker_code}.T")
            info = yft.info
            if info and len(info) > 5 and (
                info.get("currentPrice") or info.get("regularMarketPrice")
            ):
                break
        except Exception as e:
            print(f"    attempt {attempt + 1}: {e}", file=sys.stderr)

        time.sleep(1.0 * (attempt + 1))  # backoff
    else:
        return None

    # ── Price & intraday change ─────────────────────────────────────────────
    # ── Real-time price via fast_info (장중 실시간 반영) ──────────────────────
    price = None
    try:
        fi = yft.fast_info
        price = safe(getattr(fi, "last_price", None))
    except Exception:
        pass
    if price is None:
        price = safe(info.get("currentPrice")) or safe(info.get("regularMarketPrice"))

    prev_close = None
    try:
        prev_close = safe(getattr(yft.fast_info, "previous_close", None))
    except Exception:
        pass
    if prev_close is None:
        prev_close = safe(info.get("previousClose"))
    change = change_pct = None
    if price is not None and prev_close:
        change = round(price - prev_close, 2)
        change_pct = round((price / prev_close - 1) * 100, 2)

    # ── Market cap (JPY → billions) ─────────────────────────────────────────
    mcap = safe(info.get("marketCap"))
    market_cap_b = round(mcap / 1e9, 2) if mcap else None

    # ── Valuation ───────────────────────────────────────────────────────────
    per = to_ratio(info.get("trailingPE"))
    pbr = to_ratio(info.get("priceToBook"))
    div_yield = compute_div_yield(info)
    
    
    # ── Quality ─────────────────────────────────────────────────────────────
    roe = to_pct(info.get("returnOnEquity"))
    roa = to_pct(info.get("returnOnAssets"))
    op_margin = to_pct(info.get("operatingMargins"))

    # yfinance returns debtToEquity as PERCENT (105.4 means 1.054x). Our UI
    # expects the ratio.
    raw_de = info.get("debtToEquity")
    d_to_e = (
        round(float(raw_de) / 100, 2)
        if raw_de is not None and not pd.isna(raw_de)
        else None
    )

    # ── Growth ──────────────────────────────────────────────────────────────
    rev_growth = to_pct(info.get("revenueGrowth"))
    eps_growth = to_pct(info.get("earningsGrowth"))

    # ── 1Y performance ──────────────────────────────────────────────────────
    perf_1y = to_pct(info.get("52WeekChange"))

    # ── Sparkline + 20d liquidity (last ~2 months of history) ──────────────
    sparkline: list[float] = []
    adv_value_b = adv_shares_m = None
    try:
        hist = yft.history(period="2mo", auto_adjust=True)
        if not hist.empty:
            sparkline = [round(float(v), 2) for v in hist["Close"].tail(20).tolist()]
            tail = hist.tail(20)
            if "Volume" in tail and tail["Volume"].notna().any():
                vol = tail["Volume"].astype(float)
                adv_shares_m = round(float(vol.mean()) / 1e6, 3)          # M shares/day
                turnover = (tail["Close"].astype(float) * vol)
                adv_value_b = round(float(turnover.mean()) / 1e9, 3)      # ¥B/day
    except Exception as e:
        print(f"    ! history failed for {ticker_code}: {e}", file=sys.stderr)

    # ── Balance-sheet slack (for activist score) ────────────────────────────
    total_cash = safe(info.get("totalCash"))
    total_debt = safe(info.get("totalDebt"))
    net_cash_b = net_cash_pct = None
    if total_cash is not None:
        net_cash = total_cash - (total_debt or 0)
        net_cash_b = round(net_cash / 1e9, 2)
        if mcap:
            net_cash_pct = round(net_cash / mcap * 100, 1)

    # ── Share count & free float ────────────────────────────────────────────
    shares_out = safe(info.get("sharesOutstanding"))
    float_shares = safe(info.get("floatShares"))
    shares_out_m = round(shares_out / 1e6, 2) if shares_out else None
    free_float_pct = (
        round(float_shares / shares_out * 100, 1)
        if shares_out and float_shares else None
    )

    return {
        "ticker": ticker_code,
        "nameEn": meta["nameEn"],
        "nameKo": meta["nameKo"],
        "nameJp": meta["nameJp"],
        "sector": meta["sector"],
        "segment": meta["segment"],
        "price": price,
        "change": change,
        "changePercent": change_pct,
        "marketCapB": market_cap_b,
        "per": per,
        "pbr": pbr,
        "dividendYield": div_yield,
        "roe": roe,
        "roa": roa,
        "operatingMargin": op_margin,
        "debtToEquity": d_to_e,
        "revenueGrowthYoY": rev_growth,
        "epsGrowthYoY": eps_growth,
        "belowBook": pbr is not None and pbr < 1.0,
        "perf1Y": perf_1y,
        "sparkline": sparkline,
        # liquidity (computed here); short fields are attached in main()
        "advValueB": adv_value_b,
        "advSharesM": adv_shares_m,
        "sharesOutM": shares_out_m,
        "freeFloatPct": free_float_pct,
        "netCashB": net_cash_b,
        "netCashPct": net_cash_pct,
        "shortPct": None,
        "shortSharesM": None,
        "shortDtc": None,
        "shortReporters": None,
        "shortPctChange": None,
        "shortAsOf": None,
        "shortCandidate": False,
        # weekly margin balances (attached in main)
        "marginLongM": None,
        "marginShortM": None,
        "marginRatio": None,
        "marginAsOf": None,
        # sector-relative z-scores (computed in main)
        "pbrZ": None,
        "perZ": None,
        "roeZ": None,
        "valZ": None,
        # catalyst layer (attached in main)
        "catalysts": [],
        "reformDisclosed": None,
        "recentCatalyst": False,
        # activist target score (computed in main)
        "activistScore": None,
        "activistSub": None,
        "activistCandidate": False,
    }


# ============================================================================
# Index fetch
# ============================================================================
# For indices we use ETF proxies where the raw index isn't reliably available
# on Yahoo Finance.

INDEX_SPECS = [
    # symbol,       name_ko,              name_en
    ("^N225",       "닛케이 225",           "Nikkei 225"),
    ("1306.T",      "토픽스 (ETF 1306)",    "TOPIX (NEXT FUNDS ETF)"),
    ("1591.T",      "JPX 니케이 400 (ETF)", "JPX-Nikkei 400 (NEXT FUNDS ETF)"),
    ("2516.T",      "도쿄 그로스 (ETF)",    "Tokyo Growth 250 (NEXT FUNDS ETF)"),
]


def fetch_index(symbol: str, name_ko: str, name_en: str) -> dict | None:
    try:
        yft = yf.Ticker(symbol)
        info = yft.info
        hist = yft.history(period="1y", auto_adjust=True)
    except Exception as e:
        print(f"    index {symbol} failed: {e}", file=sys.stderr)
        return None

    if hist.empty:
        return None

    close = hist["Close"]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    change = last - prev
    change_pct = (last / prev - 1) * 100 if prev else 0

    high52 = safe(info.get("fiftyTwoWeekHigh"))
    low52 = safe(info.get("fiftyTwoWeekLow"))
    if high52 is None:
        high52 = float(close.max())
    if low52 is None:
        low52 = float(close.min())

    sparkline = [round(float(v), 2) for v in close.tail(20).tolist()]

    return {
        "symbol": symbol.replace("^", "").replace(".T", ""),
        "nameKo": name_ko,
        "nameEn": name_en,
        "value": round(last, 2),
        "change": round(change, 2),
        "changePercent": round(change_pct, 2),
        "high52w": round(float(high52), 2),
        "low52w": round(float(low52), 2),
        "sparkline": sparkline,
    }


# ============================================================================
# Derived aggregates
# ============================================================================

def compute_sector_performance(stocks: list[dict]) -> list[dict]:
    by_sector: dict[str, list[dict]] = defaultdict(list)
    for s in stocks:
        by_sector[s["sector"]].append(s)

    out = []
    for sector, arr in sorted(by_sector.items(), key=lambda x: -len(x[1])):
        pcts = [s["changePercent"] for s in arr if s["changePercent"] is not None]
        pbrs = [s["pbr"] for s in arr if s["pbr"] is not None]
        out.append({
            "sector": sector,
            "changePercent": round(sum(pcts) / len(pcts), 2) if pcts else 0,
            "avgPbr": round(sum(pbrs) / len(pbrs), 2) if pbrs else 0,
            "count": len(arr),
        })
    return out


def compute_market_summary(stocks: list[dict]) -> dict:
    advancers = sum(
        1 for s in stocks if s["changePercent"] is not None and s["changePercent"] > 0
    )
    decliners = sum(
        1 for s in stocks if s["changePercent"] is not None and s["changePercent"] < 0
    )
    unchanged = sum(1 for s in stocks if s["changePercent"] == 0)
    below_book = sum(1 for s in stocks if s["belowBook"])

    def avg(key: str) -> float | None:
        vals = [s[key] for s in stocks if s.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    return {
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "belowBookCount": below_book,
        "avgPer": avg("per"),
        "avgPbr": avg("pbr"),
        "avgDiv": avg("dividendYield"),
        "totalListings": len(stocks),
    }


# ============================================================================
# Short-position attachment & short-candidate screening
# ============================================================================

def attach_short_data(stocks: list[dict], short_agg: dict | None,
                      prev_stocks: dict[str, dict]) -> None:
    """Merge JPX short aggregates into stock rows (in place)."""
    if not short_agg:
        return
    by_code = short_agg.get("byCode", {})
    for s in stocks:
        agg = by_code.get(s["ticker"])
        if not agg:
            continue
        s["shortPct"] = round(agg["pct"], 2)
        s["shortReporters"] = agg["reporters"]
        s["shortAsOf"] = agg.get("latestDate")
        if agg.get("shares"):
            s["shortSharesM"] = round(agg["shares"] / 1e6, 2)
            if s.get("advSharesM"):
                s["shortDtc"] = round(s["shortSharesM"] / s["advSharesM"], 1)
        prev = prev_stocks.get(s["ticker"])
        if prev and prev.get("shortPct") is not None:
            s["shortPctChange"] = round(s["shortPct"] - prev["shortPct"], 2)


def _percentile_map(values: list[tuple[str, float]]) -> dict[str, float]:
    """ticker -> percentile rank (0-100) of value within the universe."""
    if not values:
        return {}
    ordered = sorted(values, key=lambda x: x[1])
    n = len(ordered)
    return {t: round(i / max(n - 1, 1) * 100, 1) for i, (t, _) in enumerate(ordered)}


def flag_short_candidates(stocks: list[dict]) -> int:
    """Mark liquid, expensive, fundamentally-deteriorating names.

    Criteria (all required):
      * PBR in the top 30% of the universe (rich vs peers)
      * EPS or revenue growth negative (deteriorating)
      * 1Y price performance in the bottom 40% (tape confirms)
      * ADV >= ¥1B/day (actually shortable in size)
    """
    pbr_pct = _percentile_map([(s["ticker"], s["pbr"]) for s in stocks
                               if s.get("pbr") is not None])
    perf_pct = _percentile_map([(s["ticker"], s["perf1Y"]) for s in stocks
                                if s.get("perf1Y") is not None])
    n = 0
    for s in stocks:
        t = s["ticker"]
        deteriorating = ((s.get("epsGrowthYoY") is not None and s["epsGrowthYoY"] < 0)
                         or (s.get("revenueGrowthYoY") is not None and s["revenueGrowthYoY"] < 0))
        s["shortCandidate"] = bool(
            pbr_pct.get(t, 0) >= 70
            and deteriorating
            and perf_pct.get(t, 100) <= 40
            and (s.get("advValueB") or 0) >= 1.0
        )
        n += s["shortCandidate"]
    return n


# ============================================================================
# Weekly margin balances & sector-relative z-scores
# ============================================================================

def attach_margin_data(stocks: list[dict], margin: dict | None) -> int:
    """Merge JPX weekly margin balances into stock rows (in place)."""
    if not margin:
        return 0
    by_code = margin.get("byCode", {})
    n = 0
    for s in stocks:
        m = by_code.get(s["ticker"])
        if not m:
            continue
        s["marginLongM"] = round(m["longSh"] / 1e6, 2)
        s["marginShortM"] = round(m["shortSh"] / 1e6, 2)
        s["marginRatio"] = m.get("ratio")
        s["marginAsOf"] = margin.get("asOf")
        n += 1
    return n


def compute_sector_zscores(stocks: list[dict], min_n: int = 4) -> None:
    """Sector-relative z-scores for PBR, PER and ROE, plus a composite
    valuation score valZ (mean of pbrZ/perZ; positive = rich vs sector,
    negative = cheap vs sector). Sectors with fewer than `min_n` valid
    observations are left as None.
    """
    import statistics
    by_sector: dict[str, list[dict]] = defaultdict(list)
    for s in stocks:
        by_sector[s["sector"]].append(s)

    def zmap(arr: list[dict], key: str) -> dict[str, float]:
        vals = [(s["ticker"], s[key]) for s in arr
                if s.get(key) is not None and s[key] > 0]
        if len(vals) < min_n:
            return {}
        xs = [v for _, v in vals]
        mu = statistics.fmean(xs)
        sd = statistics.pstdev(xs)
        if sd == 0:
            return {}
        return {t: round((v - mu) / sd, 2) for t, v in vals}

    for arr in by_sector.values():
        pbr_z = zmap(arr, "pbr")
        per_z = zmap(arr, "per")
        roe_z = zmap(arr, "roe")
        for s in arr:
            t = s["ticker"]
            s["pbrZ"] = pbr_z.get(t)
            s["perZ"] = per_z.get(t)
            s["roeZ"] = roe_z.get(t)
            comps = [z for z in (s["pbrZ"], s["perZ"]) if z is not None]
            s["valZ"] = round(sum(comps) / len(comps), 2) if comps else None


# ============================================================================
# Main
# ============================================================================


def compute_activist_scores(stocks: list[dict]) -> None:
    """Activist-target score (0-100): how attractive each stock looks as a
    *next* engagement target, using Japan-governance-canonical anchors so the
    number is explainable rather than a black box.

    Components (weights):
      value 30%      — PBR anchored at the TSE's 1.0x line (0.5x→full, 1.3x→0)
                       blended with the sector valuation z-score
      cash 25%       — net cash / market cap (40%+ → full). Skipped for
                       Financial Services, where balance-sheet cash isn't
                       corporate slack.
      profitability 20% — ROE gap below 8% (the Ito Review cost-of-equity
                       threshold). Loss-makers get partial credit (0.3): they
                       are turnarounds, not classic lazy-balance-sheet targets.
      payout 15%     — headroom below a 50% payout ratio, approximated as
                       divYield × PER (DPS/EPS = (DPS/P)·(P/EPS)).
      structure 10%  — free float 30%→80%: a controlling holder blocks
                       campaigns; high float means a vote can actually be won.

    Missing components are dropped and weights renormalized; a score requires
    the value component plus at least one other. Writes per stock:
      activistScore (0-100), activistSub {value,cash,prof,payout,float},
      activistCandidate (score>=70 & pbr<1.2 & advValueB>=0.5).
    """
    def clamp01(x):
        return max(0.0, min(1.0, x))

    for st in stocks:
        comps: dict[str, float] = {}

        pbr = st.get("pbr")
        if pbr is not None and pbr > 0:
            pbr_part = clamp01((1.3 - pbr) / (1.3 - 0.5))
            valz = st.get("valZ")
            if valz is not None:
                z_part = clamp01((-valz) / 2.0)        # z = -2 → full
                comps["value"] = 0.6 * pbr_part + 0.4 * z_part
            else:
                comps["value"] = pbr_part

        if st.get("sector") != "Financial Services":
            ncp = st.get("netCashPct")
            if ncp is not None:
                comps["cash"] = clamp01(ncp / 40.0)

        roe = st.get("roe")
        if roe is not None:
            comps["prof"] = clamp01((8.0 - roe) / 8.0) if roe > 0 else 0.3

        dy, per = st.get("dividendYield"), st.get("per")
        if dy is not None and per is not None and per > 0:
            payout = dy * per                          # ≈ payout ratio in %
            comps["payout"] = clamp01((50.0 - payout) / 50.0)

        ff = st.get("freeFloatPct")
        if ff is not None:
            comps["float"] = clamp01((ff - 30.0) / 50.0)

        if "value" not in comps or len(comps) < 2:
            continue

        weights = {"value": 0.30, "cash": 0.25, "prof": 0.20,
                   "payout": 0.15, "float": 0.10}
        wsum = sum(weights[k] for k in comps)
        score = sum(comps[k] * weights[k] for k in comps) / wsum * 100

        st["activistScore"] = round(score)
        st["activistSub"] = {k: round(v * 100) for k, v in comps.items()}
        st["activistCandidate"] = bool(
            score >= 70
            and pbr is not None and pbr < 1.2
            and (st.get("advValueB") or 0) >= 0.5
            # a controlling holder blocks any campaign — require a winnable float
            and (ff is None or ff >= 35.0)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="data.json", help="Path for generated data file."
    )
    parser.add_argument(
        "--tickers",
        default="scripts/tickers.json",
        help="Master ticker list (JSON array of objects).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between tickers (rate-limit).",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit to N tickers (0 = all)."
    )
    parser.add_argument(
        "--skip-jpx", action="store_true",
        help="Skip the JPX short-position feed (faster local runs).",
    )
    parser.add_argument(
        "--short-state", default="short_state.json",
        help="Path of the persistent JPX short-position state file.",
    )
    parser.add_argument(
        "--skip-catalysts", action="store_true",
        help="Skip the catalyst layer (EDINET/TDnet/JPX reform).",
    )
    parser.add_argument(
        "--force-catalysts", action="store_true",
        help="Run the catalyst scan even intraday (default: close window only).",
    )
    parser.add_argument(
        "--catalyst-state", default="catalyst_state.json",
        help="Path of the persistent catalyst state file.",
    )
    args = parser.parse_args()

    tickers_path = Path(args.tickers)
    if not tickers_path.exists():
        print(f"✗ Tickers file not found: {tickers_path}", file=sys.stderr)
        return 2

    try:
        tickers: list[dict] = json.loads(tickers_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"✗ Failed to parse {tickers_path}: {e}", file=sys.stderr)
        return 2

    if args.limit > 0:
        tickers = tickers[: args.limit]

    print(f"📦 Loaded {len(tickers)} tickers from {tickers_path}")
    print(f"🕐 Start: {datetime.now(timezone.utc).isoformat()}")
    print()

    # ── Fetch stocks ────────────────────────────────────────────────────────
    stocks: list[dict] = []
    failures: list[str] = []
    t0 = time.time()

    for i, meta in enumerate(tickers, 1):
        result = fetch_stock(meta)
        if result:
            stocks.append(result)
            tag = "✓"
        else:
            failures.append(meta["ticker"])
            tag = "✗"

        if i == 1 or i % 10 == 0 or i == len(tickers):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(tickers) - i) / rate if rate > 0 else 0
            print(
                f"  [{i:3d}/{len(tickers)}]  {tag} {meta['ticker']:<6s} "
                f"{meta['nameEn']:<32s}  {elapsed:5.0f}s elapsed, ETA {eta:.0f}s"
            )

        time.sleep(args.sleep)

    elapsed = time.time() - t0
    print()
    print(f"✓ Stocks: {len(stocks)}/{len(tickers)} in {elapsed:.0f}s "
          f"({elapsed/max(len(tickers),1):.2f}s/ticker)")
    if failures:
        print(f"⚠ Failed tickers: {', '.join(failures)}")

    # ── Fetch indices ───────────────────────────────────────────────────────
    print()
    print("📈 Fetching indices...")
    indices: list[dict] = []
    for symbol, ko, en in INDEX_SPECS:
        idx = fetch_index(symbol, ko, en)
        if idx:
            indices.append(idx)
            print(f"  ✓ {symbol:<10s} {idx['value']:>12,.2f}  ({idx['changePercent']:+.2f}%)")
        else:
            print(f"  ✗ {symbol}")

    # ── JPX short positions (best-effort) ───────────────────────────────────
    short_meta = {"ok": False, "asOf": None, "coveredStocks": 0}
    prev_stocks: dict[str, dict] = {}
    out_prev = Path(args.output)
    if out_prev.exists():
        try:
            prev_payload = json.loads(out_prev.read_text(encoding="utf-8"))
            prev_stocks = {s["ticker"]: s for s in prev_payload.get("stocks", [])}
        except Exception:
            pass

    if not args.skip_jpx:
        print()
        print("🩳 Updating JPX short-position state...")
        try:
            from jpx_short import update_short_state
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from jpx_short import update_short_state
        try:
            short_agg = update_short_state(args.short_state)
        except Exception as e:
            print(f"    ! jpx short feed failed: {e}", file=sys.stderr)
            short_agg = None
        if short_agg:
            attach_short_data(stocks, short_agg, prev_stocks)
            covered = sum(1 for s in stocks if s["shortPct"] is not None)
            short_meta = {"ok": True, "asOf": short_agg.get("asOf"),
                          "coveredStocks": covered}
            print(f"  ✓ short data attached to {covered}/{len(stocks)} stocks")
        else:
            # carry forward yesterday's values rather than blanking the UI
            for s in stocks:
                prev = prev_stocks.get(s["ticker"])
                if prev:
                    for k in ("shortPct", "shortSharesM", "shortDtc",
                              "shortReporters", "shortAsOf"):
                        s[k] = prev.get(k)
            print("  ⚠ jpx feed unavailable — carried forward previous values")

    # ── JPX weekly margin balances (best-effort) ────────────────────────────
    margin_meta = {"ok": False, "asOf": None, "coveredStocks": 0}
    if not args.skip_jpx:
        print()
        print("📊 Fetching weekly margin balances (SoftHompo)...")
        try:
            from jpx_margin import fetch_margin_balances
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from jpx_margin import fetch_margin_balances
        try:
            margin = fetch_margin_balances(
                universe={m["ticker"] for m in tickers})
        except Exception as e:
            print(f"    ! jpx margin feed failed: {e}", file=sys.stderr)
            margin = None
        if margin:
            n_m = attach_margin_data(stocks, margin)
            margin_meta = {"ok": True, "asOf": margin.get("asOf"),
                           "coveredStocks": n_m}
            print(f"  ✓ margin data attached to {n_m}/{len(stocks)} stocks")
        else:
            for s in stocks:
                prev = prev_stocks.get(s["ticker"])
                if prev:
                    for k in ("marginLongM", "marginShortM",
                              "marginRatio", "marginAsOf"):
                        s[k] = prev.get(k)
            print("  ⚠ margin feed unavailable — carried forward previous values")

    n_short = flag_short_candidates(stocks)
    print(f"  ✓ short-candidate screen: {n_short} names flagged")

    compute_sector_zscores(stocks)
    n_z = sum(1 for s in stocks if s["valZ"] is not None)
    print(f"  ✓ sector z-scores computed for {n_z}/{len(stocks)} stocks")

    compute_activist_scores(stocks)
    n_a = sum(1 for s in stocks if s["activistScore"] is not None)
    n_c = sum(1 for s in stocks if s["activistCandidate"])
    print(f"  ✓ activist scores computed for {n_a}/{len(stocks)} stocks "
          f"({n_c} candidates ≥70)")

    # ── Catalyst layer (best-effort: EDINET 5%, TDnet, JPX reform) ──────────
    # To keep intraday runs fast on a larger universe, the catalyst scan
    # (EDINET/TDnet network calls) only runs at/after the JST close unless
    # forced. The close run is the cron at UTC 7:30 (JST 16:30).
    catalyst_meta = {"edinet": False, "tdnet": False, "reform": False}
    _now_jst = datetime.now(timezone(timedelta(hours=9)))
    _is_close_window = _now_jst.hour >= 15 or args.force_catalysts
    if not args.skip_catalysts and not _is_close_window:
        print()
        print(f"⚡ Catalyst layer skipped intraday (JST {_now_jst:%H:%M}); "
              f"runs after market close. Carrying forward previous values.")
        for s in stocks:
            prev = prev_stocks.get(s["ticker"])
            if prev:
                s["catalysts"] = prev.get("catalysts", [])
                s["reformDisclosed"] = prev.get("reformDisclosed")
                s["recentCatalyst"] = prev.get("recentCatalyst", False)
    elif not args.skip_catalysts:
        print()
        print("⚡ Updating catalyst layer (EDINET / TDnet / JPX reform)...")
        try:
            from catalysts import update_catalysts
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from catalysts import update_catalysts
        try:
            cat = update_catalysts(args.catalyst_state, tickers)
        except Exception as e:
            print(f"    ! catalyst layer failed: {e}", file=sys.stderr)
            cat = None
        if cat:
            cutoff_30d = (datetime.now(timezone.utc) - timedelta(days=30)
                          ).strftime("%Y-%m-%d")
            n_attached = 0
            for s in stocks:
                evs = cat["events"].get(s["ticker"], [])
                s["catalysts"] = evs
                s["recentCatalyst"] = any(e["date"] >= cutoff_30d for e in evs)
                if s["ticker"] in cat["reform"]:
                    s["reformDisclosed"] = cat["reform"][s["ticker"]]
                if evs or s["reformDisclosed"] is not None:
                    n_attached += 1
            catalyst_meta = cat["meta"]
            print(f"  ✓ catalysts attached to {n_attached}/{len(stocks)} stocks")
        else:
            for s in stocks:
                prev = prev_stocks.get(s["ticker"])
                if prev:
                    s["catalysts"] = prev.get("catalysts", [])
                    s["reformDisclosed"] = prev.get("reformDisclosed")
                    s["recentCatalyst"] = prev.get("recentCatalyst", False)
            print("  ⚠ catalyst layer unavailable — carried forward previous values")

    # ── Derive summaries ────────────────────────────────────────────────────
    sector_perf = compute_sector_performance(stocks)
    summary = compute_market_summary(stocks)

    # ── Write data.json ─────────────────────────────────────────────────────
    _headlines_for_payload = fetch_market_headlines()
    
    payload = {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "source": "yfinance",
        "counts": {
            "stocks": len(stocks),
            "failures": len(failures),
            "indices": len(indices),
        },
        "indices": indices,
        "stocks": stocks,
        "sectorPerformance": sector_perf,
        "marketSummary": summary,
        "headlines": _headlines_for_payload,
        "marketTone": generate_market_summary(_headlines_for_payload),
        "shortData": short_meta,
        "marginData": margin_meta,
        "catalystData": catalyst_meta,
    }

    payload = clean_nans(payload)
    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    size_kb = out_path.stat().st_size / 1024
    print()
    print(f"💾 Wrote {out_path} ({size_kb:.1f} KB)")
    print(f"🕐 Done:  {datetime.now(timezone.utc).isoformat()}")

    # Fail CI if we lost too many tickers
    if len(tickers) > 0 and len(failures) > len(tickers) * 0.2:
        print(
            f"\n✗ Too many failures ({len(failures)}/{len(tickers)}, "
            f"threshold 20%)",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
