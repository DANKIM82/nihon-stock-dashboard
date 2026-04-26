"""
Fetch Japanese stock data from Yahoo Finance and write data.json.

This script is executed by GitHub Actions on a daily cron (see
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
from datetime import datetime, timezone
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

def fetch_news_for_ticker(ticker_obj, ticker_code: str, max_items: int = 2) -> list[dict]:
    """
    Pull recent news headlines from Yahoo Finance for a given ticker.
    Returns at most `max_items` items, each as a dict with title/publisher/link/published_ts.
    Failures are swallowed — news is best-effort, not critical.
    """
    try:
        raw = ticker_obj.news or []
    except Exception:
        return []

    items = []
    for n in raw[:max_items]:
        # yfinance news structure shifted around 2024 — try both shapes
        content = n.get("content") if isinstance(n, dict) else None
        if content:
            title = content.get("title")
            pub = (content.get("provider") or {}).get("displayName")
            link = (content.get("clickThroughUrl") or content.get("canonicalUrl") or {}).get("url")
            pub_date = content.get("pubDate")  # ISO string
            ts = None
            if pub_date:
                try:
                    ts = int(pd.Timestamp(pub_date).timestamp())
                except Exception:
                    ts = None
        else:
            title = n.get("title")
            pub = n.get("publisher")
            link = n.get("link")
            ts = n.get("providerPublishTime")  # already unix seconds

        if not title or not ts:
            continue
        items.append({
            "ticker": ticker_code,
            "title": title.strip(),
            "publisher": pub or "",
            "link": link or "",
            "publishedTs": int(ts),
        })
    return items

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

def collect_top_headlines(stocks: list[dict], max_total: int = 7, max_age_hours: int = 24) -> list[dict]:
    """
    Aggregate news across all stocks. Keep only the freshest 1 per ticker,
    sort by recency, drop anything older than max_age_hours, return top N.
    """
    import time
    cutoff = time.time() - max_age_hours * 3600

    seen_tickers = set()
    candidates = []
    for s in stocks:
        for n in s.get("news", []):
            if n["ticker"] in seen_tickers:
                continue
            if n["publishedTs"] < cutoff:
                continue
            candidates.append({
                **n,
                "stockName": s.get("nameEn") or s.get("nameJp") or n["ticker"],
            })
            seen_tickers.add(n["ticker"])

    candidates.sort(key=lambda x: x["publishedTs"], reverse=True)
    return candidates[:max_total]

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
    price = safe(info.get("currentPrice")) or safe(info.get("regularMarketPrice"))
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
    news_items = fetch_news_for_ticker(yft, ticker_code)
    
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

    # ── Sparkline (last 20 trading days) ────────────────────────────────────
    sparkline: list[float] = []
    try:
        hist = yft.history(period="1mo", auto_adjust=True)
        if not hist.empty:
            sparkline = [round(float(v), 2) for v in hist["Close"].tail(20).tolist()]
    except Exception as e:
        print(f"    ! history failed for {ticker_code}: {e}", file=sys.stderr)

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
        "news": news_items,
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
# Main
# ============================================================================

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

    # ── Derive summaries ────────────────────────────────────────────────────
    sector_perf = compute_sector_performance(stocks)
    summary = compute_market_summary(stocks)

    # ── Write data.json ─────────────────────────────────────────────────────
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
        "headlines": collect_top_headlines(stocks),
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
