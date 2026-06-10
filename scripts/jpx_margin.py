"""
JPX 銘柄別信用取引週末残高 (per-stock weekly margin balances).

JPX publishes, once a week, an Excel file with every margin-tradable stock's
outstanding margin BUY balance (買残高 — retail leverage longs) and margin
SELL balance (売残高 — margin shorts), in shares. Unlike the short-position
reports, each weekly file is a FULL SNAPSHOT, so no accumulation state is
needed — we simply parse the most recent file.

Derived metric: 信用倍率 (margin ratio) = 買残 / 売残. A high ratio means
crowded retail longs (overhead supply); a ratio < 1 means margin shorts
outnumber margin longs (squeeze fuel).

Public API:
    fetch_margin_balances() -> dict | None
        {"byCode": {code: {"longSh": int, "shortSh": int, "ratio": float}},
         "asOf": "YYYY-MM-DD" | None}

Best-effort: returns None on any failure, never raises.
"""

from __future__ import annotations

import io
import re
import sys
import time
import unicodedata
import urllib.request

import pandas as pd

JPX_MARGIN_PAGES = [
    "https://www.jpx.co.jp/markets/statistics-equities/margin/index.html",
    "https://www.jpx.co.jp/markets/statistics-equities/margin/05-01.html",
]
JPX_BASE = "https://www.jpx.co.jp"
UA = {"User-Agent": "Mozilla/5.0 (nihon-dashboard data fetcher)"}
MAX_TRIES = 6   # candidate files to attempt before giving up


def _get(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return unicodedata.normalize("NFKC", str(s)).replace("\n", "").replace(" ", "")


def _list_candidate_urls() -> list[str]:
    urls, seen = [], set()
    for page in JPX_MARGIN_PAGES:
        try:
            html = _get(page).decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"    ! jpx margin index fetch failed ({page}): {e}", file=sys.stderr)
            continue
        for m in re.finditer(r'href="([^"]+\.(?:xls|xlsx))"', html, re.I):
            href = m.group(1)
            url = href if href.startswith("http") else JPX_BASE + href
            if url in seen:
                continue
            seen.add(url)
            # the per-stock weekly file is usually named like "syumatsu..." (週末)
            score = 0 if "syumatsu" in url.lower() else 1
            urls.append((score, url))
    urls.sort(key=lambda x: x[0])
    return [u for _, u in urls]


def _merged_headers(df: pd.DataFrame, hi: int, depth: int = 3) -> list[str]:
    """JPX margin files use multi-row headers (売残高 split into 制度/一般/合計).
    Concatenate up to `depth` header rows per column so we can match on the
    combined text, with forward-fill across columns for merged cells."""
    ncols = df.shape[1]
    out = [""] * ncols
    for r in range(hi, min(hi + depth, len(df))):
        last = ""
        for c in range(ncols):
            v = _norm(df.iat[r, c])
            if v:
                last = v
            # forward-fill only on the top (merged) row
            out[c] += v if (v or r > hi) else last
    return out


def _parse_int(v) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = _norm(v).replace(",", "").replace("株", "").split(".")[0]
    if s in ("", "-", "—", "ー"):
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _find_asof(df: pd.DataFrame) -> str | None:
    for i in range(min(8, len(df))):
        for c in range(min(8, df.shape[1])):
            s = _norm(df.iat[i, c])
            m = re.search(r"(\d{4})[年/\-.](\d{1,2})[月/\-.](\d{1,2})", s)
            if m:
                return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def _parse_margin_file(content: bytes, url: str) -> tuple[dict, str | None] | None:
    try:
        engine = "xlrd" if url.lower().endswith(".xls") else "openpyxl"
        sheets = pd.read_excel(io.BytesIO(content), sheet_name=None,
                               header=None, engine=engine)
    except Exception as e:
        print(f"    ! margin parse failed {url.rsplit('/',1)[-1]}: {e}", file=sys.stderr)
        return None

    by_code: dict[str, dict] = {}
    asof = None
    for _, df in sheets.items():
        if df is None or df.empty or df.shape[1] < 4:
            continue
        if asof is None:
            asof = _find_asof(df)
        # locate header row containing コード
        hi = None
        for i in range(min(15, len(df))):
            row = "|".join(_norm(v) for v in df.iloc[i].tolist())
            if "コード" in row or "Code" in row:
                hi = i
                break
        if hi is None:
            continue
        headers = _merged_headers(df, hi)

        def col(*needles, prefer="合計"):
            cands = [j for j, h in enumerate(headers)
                     if all(n in h for n in needles)]
            if not cands:
                return None
            for j in cands:
                if prefer in headers[j]:
                    return j
            return cands[-1]

        c_code = next((j for j, h in enumerate(headers) if "コード" in h or "Code" in h), None)
        c_sell = col("売", "残")
        c_buy = col("買", "残")
        if c_code is None or c_sell is None or c_buy is None:
            continue

        for i in range(hi + 1, len(df)):
            code = _norm(df.iat[i, c_code]).split(".")[0]
            # JPX code might be 5-digit, so try both 4 and 5 digit formats
            if len(code) >= 4 and code[:4].isdigit():
                code4 = code[:4]
            elif len(code) >= 5 and code[:5].isdigit():
                code4 = code[:5]
            else:
                continue
            entry = by_code.setdefault(code4, {"longSh": 0, "shortSh": 0})
            entry["longSh"] += buy or 0
            entry["shortSh"] += sell or 0

    if not by_code:
        return None
    for v in by_code.values():
        v["ratio"] = round(v["longSh"] / v["shortSh"], 2) if v["shortSh"] else None
    return by_code, asof


def fetch_margin_balances(sleep: float = 0.5) -> dict | None:
    urls = _list_candidate_urls()
    print(f"    jpx margin: {len(urls)} candidate files linked")
    for url in urls[:MAX_TRIES]:
        try:
            content = _get(url)
        except Exception as e:
            print(f"    ! margin download failed {url.rsplit('/',1)[-1]}: {e}",
                  file=sys.stderr)
            continue
        parsed = _parse_margin_file(content, url)
        time.sleep(sleep)
        if parsed:
            by_code, asof = parsed
            # DEBUG: 파싱된 종목코드 출력
            parsed_codes = list(by_code.keys())
            print(f"    DEBUG parsed codes (first 20): {parsed_codes[:20]}", file=sys.stderr)
            print(f"    jpx margin: {len(by_code)} stocks parsed from "
                  f"{url.rsplit('/',1)[-1]} (as of {asof})")
            return {"byCode": by_code, "asOf": asof}            
    print("    ! jpx margin: no parsable file found", file=sys.stderr)
    return None
