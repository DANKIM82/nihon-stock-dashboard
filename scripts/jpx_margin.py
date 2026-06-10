"""
銘柄別信用取引週末残高 (per-stock weekly margin balances) via SoftHompo.

JPX's own free pages only expose the *daily-publication* file (a few hundred
flagged small-caps) — the full all-stocks weekly balance is paid data. SoftHompo
(softhompo.a.la9.jp) republishes the TSE weekly margin-balance report for ALL
issues as a zipped CSV, refreshed the 2nd business day of each week. We pull the
latest weekly file, parse it, and attach per-stock margin long/short balances.

This is a community-run mirror, not an official feed: it can disappear or lag.
The fetch is best-effort and the caller carries forward previous values on
failure. Coverage and 信用倍率 (margin ratio = long / short) are exposed.

Public API (unchanged from the previous JPX version):
    fetch_margin_balances(sleep=..., universe=set[str]|None) -> dict | None
        {"byCode": {code: {"longSh": int, "shortSh": int, "ratio": float}},
         "asOf": "YYYY-MM-DD" | None}

The CSV column layout is auto-detected from the header comment lines, so small
changes in column order are tolerated. If detection fails, a fallback index map
(derived from the documented TSE weekly format) is used.
"""

from __future__ import annotations

import io
import re
import sys
import time
import unicodedata
import urllib.request
import zipfile

import pandas as pd

PAGE_URL = "https://softhompo.a.la9.jp/Data/StockData.html"
SITE_BASE = "https://softhompo.a.la9.jp"
UA = {"User-Agent": "Mozilla/5.0 (nihon-dashboard data fetcher)"}
MAX_TRIES = 4   # candidate zips to attempt (newest first)


# ──────────────────────────────────────────────────────────────────────────
# HTTP / code helpers
# ──────────────────────────────────────────────────────────────────────────
def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _norm(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return unicodedata.normalize("NFKC", str(s)).replace("\n", "").replace(" ", "")


def _canon_code(raw: str) -> str | None:
    """Normalize a JPX security code to the 4-char form the dashboard uses.

    JPX/TSE express ordinary shares as a 5-digit code with a trailing 0
    (Toyota = '72030'); the UI universe uses 4-digit ('7203'). New-style codes
    are 4-char alphanumerics ending in a letter ('278A') and are kept as-is.
        '72030' -> '7203'      '7203'  -> '7203'
        '278A0' -> '278A'      '278A'  -> '278A'
    """
    c = _norm(raw).split(".")[0].upper()
    if re.fullmatch(r"\d{4}", c):
        return c
    if re.fullmatch(r"\d{3}[A-Z]", c):
        return c
    if re.fullmatch(r"\d{5}", c) and c.endswith("0"):
        return c[:4]
    if re.fullmatch(r"\d{3}[A-Z]0", c):
        return c[:4]
    if re.fullmatch(r"\d{4}[A-Z0-9]", c):
        return c[:4]
    return None


def _parse_int(v) -> int | None:
    if v is None:
        return None
    s = _norm(v).replace(",", "").replace("株", "")
    s = s.split(".")[0]
    if s in ("", "-", "—", "ー", "*"):
        return None
    m = re.match(r"-?\d+", s)
    return int(m.group()) if m else None


# ──────────────────────────────────────────────────────────────────────────
# Locate the latest weekly zip
# ──────────────────────────────────────────────────────────────────────────
def _list_margin_zips() -> list[str]:
    try:
        html = _get(PAGE_URL).decode("utf-8", "ignore")
    except Exception as e:
        print(f"    ! softhompo page fetch failed: {e}", file=sys.stderr)
        return []
    # The page hrefs are abbreviated (e.g. '/thisMonth/syumatsuYYYYMMDD00.zip'),
    # but the files actually live under '/Data/margin/{thisMonth,pastMonth}/'.
    # Extract just the filename and rebuild the full path to be safe.
    this_fn = re.findall(r'syumatsu(\d+)\.zip', html)
    past_fn = re.findall(r'/(?:pastMonth|margin/pastMonth)/(\d{6})\.zip', html)
    dl = "https://softhompo.a.la9.jp/Data/margin"
    ordered = [f"{dl}/thisMonth/syumatsu{d}.zip"
               for d in sorted(set(this_fn), reverse=True)]
    ordered += [f"{dl}/pastMonth/{m}.zip"
                for m in sorted(set(past_fn), reverse=True)]
    return ordered


# ──────────────────────────────────────────────────────────────────────────
# CSV parsing with auto column detection
# ──────────────────────────────────────────────────────────────────────────
# Confirmed SoftHompo CSV layout (verified against the real 2026-06-05 file).
# Header is 3 comment rows starting with '!'; data rows look like:
#   col0=売買単位  col1=銘柄名  col2=コード(4-digit)  col3=新証券コード
#   col4=売残高合計(short)  col5=前週比  col6=買残高合計(long)  col7=前週比
#   col8..15 = 一般/制度 internal splits (ignored)
_COLS = {"code": 2, "sell": 4, "buy": 6, "date_in_header": True}


def _find_asof(lines: list[str]) -> str | None:
    # row index 3 in the real file is: "!","2026/6/5","",""
    for ln in lines[:8]:
        m = re.search(r"(\d{4})[/\-年.](\d{1,2})[/\-月.](\d{1,2})", ln)
        if m:
            return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def _split_csv(line: str) -> list[str]:
    """Split a CSV line, stripping surrounding double-quotes from each cell.
    SoftHompo quotes text fields but not numbers; a simple split on commas
    works because there are no embedded commas inside the quoted names."""
    return [c.strip().strip('"').strip() for c in line.split(",")]


def _decode(data: bytes) -> str | None:
    for enc in ("utf-8-sig", "utf-8", "cp932", "shift_jis", "euc-jp"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return None


def _parse_margin_csv(text: str) -> tuple[dict, str | None] | None:
    lines = text.splitlines()
    if not lines:
        return None
    asof = _find_asof(lines)

    ci, si, bi = _COLS["code"], _COLS["sell"], _COLS["buy"]
    need = max(ci, si, bi) + 1

    by_code: dict[str, dict] = {}
    for ln in lines:
        if not ln or ln.lstrip().startswith('"!') or ln.lstrip().startswith("!"):
            continue
        cells = _split_csv(ln)
        if len(cells) < need:
            continue
        code4 = _canon_code(cells[ci])
        if code4 is None:
            continue
        sell = _parse_int(cells[si])   # 売残高 = margin short
        buy = _parse_int(cells[bi])    # 買残高 = margin long
        if sell is None and buy is None:
            continue
        entry = by_code.setdefault(code4, {"longSh": 0, "shortSh": 0})
        entry["longSh"] += buy or 0
        entry["shortSh"] += sell or 0

    if not by_code:
        return None
    for v in by_code.values():
        v["ratio"] = round(v["longSh"] / v["shortSh"], 2) if v["shortSh"] else None
    return by_code, asof


def _parse_zip(raw: bytes) -> tuple[dict, str | None] | None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
    except Exception as e:
        print(f"    ! margin zip invalid: {e}", file=sys.stderr)
        return None
    # a monthly archive holds several daily CSVs — parse the LATEST (max name)
    members = [n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))]
    members = members or zf.namelist()
    if not members:
        return None
    member = sorted(members)[-1]
    text = _decode(zf.read(member))
    if text is None:
        print(f"    ! could not decode {member}", file=sys.stderr)
        return None
    return _parse_margin_csv(text)


# ──────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────
def fetch_margin_balances(sleep: float = 0.5,
                          universe: set[str] | None = None) -> dict | None:
    """Download & parse the latest all-stocks weekly margin-balance file.

    When `universe` is given, a parsed file is accepted only if it overlaps the
    universe (guards against a malformed/empty file); the best candidate is kept
    as a fallback. Best-effort: returns None on total failure, never raises.
    """
    zips = _list_margin_zips()
    print(f"    margin: {len(zips)} weekly zip(s) linked on SoftHompo")
    if not zips:
        return None

    best = None
    for url in zips[:MAX_TRIES]:
        name = url.rsplit("/", 1)[-1]
        try:
            raw = _get(url)
        except Exception as e:
            print(f"    ! margin download failed {name}: {e}", file=sys.stderr)
            continue
        parsed = _parse_zip(raw)
        time.sleep(sleep)
        if not parsed:
            continue
        by_code, asof = parsed
        overlap = len(universe & set(by_code)) if universe else len(by_code)
        print(f"    margin: {len(by_code)} stocks from {name} (as of {asof}) "
              f"— universe overlap {overlap}"
              f"{'/' + str(len(universe)) if universe else ''}")
        if universe and overlap >= max(20, len(universe) // 4):
            return {"byCode": by_code, "asOf": asof}
        if universe is None and len(by_code) >= 1000:
            return {"byCode": by_code, "asOf": asof}
        if best is None or overlap > best[0]:
            best = (overlap, {"byCode": by_code, "asOf": asof}, name)

    if best and best[0] > 0:
        print(f"    margin: using best available '{best[2]}' (overlap {best[0]})",
              file=sys.stderr)
        return best[1]
    print("    ! margin: no usable weekly file found", file=sys.stderr)
    return None
