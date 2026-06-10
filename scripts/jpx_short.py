"""
JPX 空売り残高 (short-position) accumulator.

JPX publishes a daily Excel file listing short positions ≥ 0.5% of shares
outstanding (and one final report when a position drops below 0.5%).
Crucially, each daily file contains ONLY that day's submissions — it is a
delta, not a snapshot. To know the current outstanding short balance per
stock we must accumulate the latest report per (reporter, ticker) pair and
drop pairs whose latest report fell below the 0.5% threshold.

State is persisted to `short_state.json` (committed by the GitHub Actions
workflow) so each run only processes files it hasn't seen before.

Known caveats (by design of the disclosure regime):
  * Only positions ≥ 0.5% are visible → the aggregate is a LOWER BOUND of
    true short interest.
  * On first bootstrap we only see files still linked on the JPX page
    (~recent weeks), so long-standing static positions reported earlier
    are missed until the reporter next updates. Coverage converges over
    time.

Public API:
    update_short_state(state_path) -> dict   # {"byCode": {...}, "asOf": ...}

Each byCode entry:
    {"pct": 1.83, "shares": 12_345_600, "reporters": 4, "latestDate": "2026-06-08"}
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

JPX_INDEX_URLS = [
    "https://www.jpx.co.jp/markets/public/short-selling/index.html",
    # archive page sometimes holds the tail end of the current month
    "https://www.jpx.co.jp/markets/public/short-selling/00-archives-01.html",
]
JPX_BASE = "https://www.jpx.co.jp"
UA = {"User-Agent": "Mozilla/5.0 (nihon-dashboard data fetcher)"}

MAX_FILES_PER_RUN = 40          # bootstrap safety valve
THRESHOLD_PCT = 0.5             # disclosure threshold


# ──────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────
def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_report_urls() -> list[str]:
    """Scrape the JPX short-selling pages for daily report Excel links."""
    urls: list[str] = []
    seen = set()
    for page in JPX_INDEX_URLS:
        try:
            html = _get(page).decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"    ! jpx index fetch failed ({page}): {e}", file=sys.stderr)
            continue
        for m in re.finditer(r'href="([^"]+\.(?:xls|xlsx|csv))"', html, re.I):
            href = m.group(1)
            url = href if href.startswith("http") else JPX_BASE + href
            if url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


# ──────────────────────────────────────────────────────────────────────────
# Excel parsing
# ──────────────────────────────────────────────────────────────────────────
def _norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return unicodedata.normalize("NFKC", str(s)).replace("\n", "").replace(" ", "")


def _find_header_row(df: pd.DataFrame) -> int | None:
    for i in range(min(12, len(df))):
        row = [_norm(v) for v in df.iloc[i].tolist()]
        joined = "|".join(row)
        if ("銘柄コード" in joined or "コード" in joined) and ("割合" in joined or "Ratio" in joined):
            return i
    return None


def _col_idx(headers: list[str], *keys: str) -> int | None:
    for j, h in enumerate(headers):
        for k in keys:
            if k in h:
                return j
    return None


def _parse_pct(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = _norm(v).replace("%", "").replace(",", "")
    try:
        x = float(s)
    except ValueError:
        return None
    # files store either 0.0183 (decimal) or 1.83 (percent)
    if 0 < x <= 0.2:
        x *= 100
    return round(x, 4)


def _parse_int(v) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = _norm(v).replace(",", "").split(".")[0]
    try:
        return int(s)
    except ValueError:
        return None


def _parse_date(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    s = _norm(v)
    m = re.match(r"(\d{4})[年/\-.](\d{1,2})[月/\-.](\d{1,2})", s)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"^(\d{8})$", s)
    if m:
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None


def parse_report(content: bytes, url: str) -> list[dict]:
    """Parse one daily JPX report into rows of
    {date, reporter, code, pct, shares}."""
    rows: list[dict] = []
    try:
        if url.lower().endswith(".csv"):
            raw = pd.read_csv(io.BytesIO(content), header=None, dtype=str,
                              encoding="cp932", on_bad_lines="skip")
            sheets = {"csv": raw}
        else:
            engine = "xlrd" if url.lower().endswith(".xls") else "openpyxl"
            sheets = pd.read_excel(io.BytesIO(content), sheet_name=None,
                                   header=None, engine=engine)
    except Exception as e:
        print(f"    ! parse failed {url.rsplit('/',1)[-1]}: {e}", file=sys.stderr)
        return rows

    for _, df in sheets.items():
        if df is None or df.empty:
            continue
        hi = _find_header_row(df)
        if hi is None:
            continue
        headers = [_norm(v) for v in df.iloc[hi].tolist()]
        c_date = _col_idx(headers, "計算年月日", "年月日", "Date")
        c_rep = _col_idx(headers, "商号", "名称", "氏名", "Name")
        c_code = _col_idx(headers, "銘柄コード", "コード", "Code")
        c_pct = _col_idx(headers, "残高割合", "割合", "Ratio")
        c_qty = _col_idx(headers, "残高数量", "数量", "Number")
        if c_code is None or c_pct is None:
            continue

        for i in range(hi + 1, len(df)):
            r = df.iloc[i]
            code = _norm(r.iloc[c_code]) if c_code is not None else ""
            code = code.split(".")[0]
            if not re.match(r"^\d{4,5}[A-Z]?$", code):
                continue
            pct = _parse_pct(r.iloc[c_pct])
            if pct is None:
                continue
            rows.append({
                "date": _parse_date(r.iloc[c_date]) if c_date is not None else None,
                "reporter": _norm(r.iloc[c_rep])[:80] if c_rep is not None else "?",
                "code": code[:4],   # UI uses 4-digit codes
                "pct": pct,
                "shares": _parse_int(r.iloc[c_qty]) if c_qty is not None else None,
            })
    return rows


# ──────────────────────────────────────────────────────────────────────────
# State accumulation
# ──────────────────────────────────────────────────────────────────────────
def update_short_state(state_path: str | Path,
                       sleep: float = 0.5) -> dict | None:
    """Fetch unseen JPX reports, fold into state, return aggregate per code.

    Returns None if nothing could be fetched AND no prior state exists.
    Never raises — this feed is best-effort.
    """
    state_path = Path(state_path)
    state = {"processedUrls": [], "positions": {}}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"    ! short_state unreadable, rebuilding: {e}", file=sys.stderr)

    processed = set(state.get("processedUrls", []))
    positions: dict[str, dict] = state.get("positions", {})

    urls = list_report_urls()
    fresh = [u for u in urls if u not in processed][:MAX_FILES_PER_RUN]
    print(f"    jpx: {len(urls)} files linked, {len(fresh)} new")

    n_rows = 0
    for url in fresh:
        try:
            content = _get(url)
        except Exception as e:
            print(f"    ! jpx download failed {url.rsplit('/',1)[-1]}: {e}",
                  file=sys.stderr)
            continue
        rows = parse_report(content, url)
        n_rows += len(rows)
        for row in rows:
            key = f"{row['code']}|{row['reporter']}"
            prev = positions.get(key)
            # keep only the most recent report per (code, reporter)
            if prev and prev.get("date") and row.get("date") and prev["date"] > row["date"]:
                continue
            positions[key] = {
                "code": row["code"], "pct": row["pct"],
                "shares": row["shares"], "date": row["date"],
            }
        processed.add(url)
        time.sleep(sleep)

    if not positions:
        return None

    # drop closed positions (final sub-threshold report)
    positions = {k: v for k, v in positions.items()
                 if v.get("pct") is not None and v["pct"] >= THRESHOLD_PCT}

    # aggregate per code
    by_code: dict[str, dict] = {}
    for v in positions.values():
        agg = by_code.setdefault(v["code"], {"pct": 0.0, "shares": 0,
                                             "reporters": 0, "latestDate": None})
        agg["pct"] = round(agg["pct"] + v["pct"], 3)
        agg["shares"] += v["shares"] or 0
        agg["reporters"] += 1
        if v.get("date") and (agg["latestDate"] is None or v["date"] > agg["latestDate"]):
            agg["latestDate"] = v["date"]

    # persist (cap processedUrls so the file doesn't grow forever)
    state_out = {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "processedUrls": sorted(processed)[-400:],
        "positions": positions,
    }
    try:
        state_path.write_text(json.dumps(state_out, ensure_ascii=False),
                              encoding="utf-8")
    except Exception as e:
        print(f"    ! short_state write failed: {e}", file=sys.stderr)

    print(f"    jpx: parsed {n_rows} rows → {len(positions)} live positions "
          f"across {len(by_code)} stocks")
    return {"byCode": by_code,
            "asOf": max((v["latestDate"] or "" for v in by_code.values()),
                        default=None)}
