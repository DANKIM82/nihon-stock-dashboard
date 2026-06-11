"""
Catalyst layer: per-stock event timeline from three sources.

1) EDINET (official FSA API, requires EDINET_API_KEY env var)
   大量保有報告書 / 変更報告書 (5% large-shareholding reports) — detects
   activist/fund entries. Documented API; the daily document list gives the
   filer name and the issuer's EDINET code, which we map to our tickers via
   the EDINET code master (downloaded once, cached in state).

2) TDnet daily timely-disclosure list (HTML scrape of release.tdnet.info)
   Keyword-matched events: buyback (自己株式), tender offer (公開買付/TOB),
   MBO, capital-cost/reform disclosures (資本コスト), guidance revisions
   (業績予想の修正), dividend revisions (配当予想の修正).

3) JPX capital-efficiency disclosure status list (monthly xlsx)
   Tags each company as having disclosed (or not) its 「資本コストや株価を
   意識した経営」 plan — turns the below-book flag into an actionable
   pressure signal.

All three are best-effort and independent: any can fail without breaking the
run. Events accumulate in catalyst_state.json (rolling window), since daily
scans only see that day's filings.

Public API:
    update_catalysts(state_path, universe_meta) -> dict
        universe_meta: list of {"ticker": ..., "nameJp": ...} from tickers.json
        returns {"events": {ticker: [event, ...]},
                 "reform": {ticker: bool},
                 "meta": {...per-source ok flags...}}

Event shape:
    {"date": "YYYY-MM-DD", "type": "5pct|buyback|tob|mbo|reform|guidance|dividend",
     "title": str, "filer": str|None, "source": "edinet|tdnet"}
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (nihon-dashboard data fetcher)"}

# Optional SSL bypass for corporate networks that MITM the TLS chain. Off by
# default and unnecessary on GitHub Actions; enable locally only if you hit
# CERTIFICATE_VERIFY_FAILED, via:  set CATALYST_INSECURE_SSL=1
import ssl as _ssl
if os.environ.get("CATALYST_INSECURE_SSL") == "1":
    _SSL_CTX = _ssl.create_default_context()
    _SSL_CTX.check_hostname = False
    _SSL_CTX.verify_mode = _ssl.CERT_NONE
else:
    _SSL_CTX = None

EDINET_DOCS_URL = "https://api.edinet-fsa.go.jp/api/v2/documents.json"
EDINET_CODELIST_URL = ("https://disclosure2dl.edinet-fsa.go.jp/searchdocument/"
                       "codelist/Edinetcode.zip")
# 大量保有報告書 / 変更報告書 document type codes in the EDINET list API
EDINET_5PCT_TYPES = {"350": "신규", "360": "변경"}

TDNET_LIST_URL = "https://www.release.tdnet.info/inbs/I_list_{page:03d}_{ymd}.html"
TDNET_MAX_PAGES = 12

JPX_REFORM_PAGES = [
    "https://www.jpx.co.jp/equities/follow-up/index.html",
    "https://www.jpx.co.jp/english/equities/follow-up/index.html",
]
JPX_BASE = "https://www.jpx.co.jp"

EVENT_WINDOW_DAYS = 90
MAX_EVENTS_PER_STOCK = 12
BACKFILL_DAYS = 7          # scan up to N recent unprocessed days per run

# TDnet title keywords → event type (first match wins)
TDNET_KEYWORDS = [
    ("MBO", "mbo"),
    ("公開買付", "tob"),
    ("自己株式", "buyback"),
    ("資本コスト", "reform"),
    ("業績予想の修正", "guidance"),
    ("配当予想の修正", "dividend"),
]


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def _get(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        return r.read()


def _norm(s) -> str:
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", str(s)).strip()


def _canon_code(raw) -> str | None:
    """JPX 5-digit ('72030') or 4-char codes → the dashboard's 4-char form."""
    c = _norm(raw).split(".")[0].upper().replace('"', "")
    if re.fullmatch(r"\d{4}", c) or re.fullmatch(r"\d{3}[A-Z]", c):
        return c
    if re.fullmatch(r"\d{5}", c) and c.endswith("0"):
        return c[:4]
    if re.fullmatch(r"\d{3}[A-Z]0", c):
        return c[:4]
    if re.fullmatch(r"\d{4}[A-Z0-9]", c):
        return c[:4]
    return None


def _jst_today() -> datetime:
    return datetime.now(timezone(timedelta(hours=9)))


# ──────────────────────────────────────────────────────────────────────────
# 1) EDINET — 5% large-shareholding reports
# ──────────────────────────────────────────────────────────────────────────
def _load_edinet_map(state: dict, universe: set[str]) -> dict[str, str]:
    """Return {issuerEdinetCode: ticker} restricted to our universe.
    Built once from the EDINET code master and cached in state; rebuilt if
    the cached map is missing tickers (universe changed)."""
    cached = state.get("edinetMap", {})
    if cached and set(cached.values()) >= universe & set(cached.values()):
        covered = len(set(cached.values()) & universe)
        if covered >= len(universe) * 0.8:
            return cached

    print("    catalysts: downloading EDINET code master...")
    try:
        raw = _get(EDINET_CODELIST_URL, timeout=90)
        zf = zipfile.ZipFile(io.BytesIO(raw))
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(member).decode("cp932", "ignore")
    except Exception as e:
        print(f"    ! edinet code master failed: {e}", file=sys.stderr)
        return cached  # keep whatever we had

    mapping: dict[str, str] = {}
    lines = text.splitlines()
    # header row is the 2nd line; locate ＥＤＩＮＥＴコード & 証券コード columns
    header_idx, e_col, s_col = None, None, None
    for i, ln in enumerate(lines[:5]):
        cells = [_norm(c).strip('"') for c in ln.split(",")]
        for j, c in enumerate(cells):
            cu = unicodedata.normalize("NFKC", c).upper()
            if "EDINET" in cu:
                header_idx, e_col = i, j
            if "証券コード" in unicodedata.normalize("NFKC", c):
                s_col = j
        if e_col is not None and s_col is not None:
            break
    if e_col is None or s_col is None:
        print("    ! edinet code master: header not found", file=sys.stderr)
        return cached

    for ln in lines[header_idx + 1:]:
        cells = [c.strip('"').strip() for c in ln.split(",")]
        if len(cells) <= max(e_col, s_col):
            continue
        ecode = cells[e_col]
        ticker = _canon_code(cells[s_col])
        if ecode and ticker and ticker in universe:
            mapping[ecode] = ticker

    print(f"    catalysts: edinet map built for {len(mapping)}/{len(universe)} tickers")
    state["edinetMap"] = mapping
    return mapping


def _scan_edinet_day(ymd_dash: str, api_key: str,
                     emap: dict[str, str]) -> list[tuple[str, dict]]:
    """One day's document list → [(ticker, event), ...] for 5% reports."""
    url = (f"{EDINET_DOCS_URL}?date={ymd_dash}&type=2"
           f"&Subscription-Key={api_key}")
    try:
        data = json.loads(_get(url).decode("utf-8"))
    except Exception as e:
        print(f"    ! edinet {ymd_dash} failed: {e}", file=sys.stderr)
        return []
    out = []
    for doc in data.get("results", []) or []:
        dtype = str(doc.get("docTypeCode") or "")
        if dtype not in EDINET_5PCT_TYPES:
            continue
        desc = _norm(doc.get("docDescription"))
        if "訂正" in desc:            # corrections are noise, skip
            continue
        issuer = doc.get("issuerEdinetCode") or doc.get("subjectEdinetCode")
        ticker = emap.get(issuer or "")
        if not ticker:
            continue
        is_new = dtype == "350"
        out.append((ticker, {
            "date": ymd_dash,
            "type": "5pct",
            "title": f"大量保有報告書 ({EDINET_5PCT_TYPES[dtype]})",
            "filer": _norm(doc.get("filerName"))[:60] or None,
            "source": "edinet",
        }))
    return out


# ──────────────────────────────────────────────────────────────────────────
# 2) TDnet — timely disclosures (buybacks, TOB/MBO, reform, guidance)
# ──────────────────────────────────────────────────────────────────────────
# Rows on the list pages carry cells with classes kjTime/kjCode/kjName/kjTitle.
_TD_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TD_CELL_RE = re.compile(
    r'class="[^"]*kj(Time|Code|Name|Title)[^"]*"[^>]*>(.*?)</td>', re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _scan_tdnet_day(ymd: str) -> list[tuple[str, dict]]:
    """One day's TDnet list pages → [(ticker, event), ...] via title keywords.
    ymd: 'YYYYMMDD'."""
    out = []
    for page in range(1, TDNET_MAX_PAGES + 1):
        url = TDNET_LIST_URL.format(page=page, ymd=ymd)
        try:
            html = _get(url).decode("utf-8", "ignore")
        except Exception:
            break  # past the last page (404) or day not present
        rows = _TD_ROW_RE.findall(html)
        found_any = False
        for row in rows:
            cells = {m.group(1): _TAG_RE.sub("", m.group(2)).strip()
                     for m in _TD_CELL_RE.finditer(row)}
            code = _canon_code(cells.get("Code", ""))
            title = _norm(cells.get("Title", ""))
            if not code or not title:
                continue
            found_any = True
            for kw, etype in TDNET_KEYWORDS:
                if kw in title:
                    out.append((code, {
                        "date": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]}",
                        "type": etype,
                        "title": title[:90],
                        "filer": None,
                        "source": "tdnet",
                    }))
                    break
        if not found_any and page > 1:
            break
        time.sleep(0.3)
    return out


# ──────────────────────────────────────────────────────────────────────────
# 3) JPX capital-efficiency disclosure status (monthly list)
# ──────────────────────────────────────────────────────────────────────────
def fetch_reform_status(universe: set[str]) -> dict[str, bool] | None:
    """Parse JPX's 開示状況一覧 xlsx → {ticker: disclosed?}. Best-effort."""
    try:
        import pandas as pd
    except ImportError:
        return None

    xlsx_urls = []
    for page in JPX_REFORM_PAGES:
        try:
            html = _get(page).decode("utf-8", "ignore")
        except Exception as e:
            print(f"    ! jpx reform page failed ({page}): {e}", file=sys.stderr)
            continue
        for m in re.finditer(r'href="([^"]+\.xlsx?)"', html, re.I):
            href = m.group(1)
            url = href if href.startswith("http") else JPX_BASE + href
            if url not in xlsx_urls:
                xlsx_urls.append(url)
    print(f"    catalysts: jpx reform page lists {len(xlsx_urls)} xlsx file(s)")

    for url in xlsx_urls[:5]:
        try:
            raw = _get(url, timeout=90)
            engine = "xlrd" if url.lower().endswith(".xls") else "openpyxl"
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None,
                                   header=None, engine=engine)
        except Exception as e:
            print(f"    ! reform xlsx failed {url.rsplit('/',1)[-1]}: {e}",
                  file=sys.stderr)
            continue
        status: dict[str, bool] = {}
        for _, df in sheets.items():
            if df is None or df.empty:
                continue
            # find code column = the one with the most code-like values
            best_col, best_hits = None, 0
            for j in range(min(6, df.shape[1])):
                hits = sum(1 for v in df.iloc[:, j]
                           if _canon_code(v) is not None)
                if hits > best_hits:
                    best_col, best_hits = j, hits
            if best_col is None or best_hits < 100:
                continue
            # disclosure column: contains 開示 / 検討 markers
            disc_col = None
            for j in range(df.shape[1]):
                col_text = "".join(_norm(v) for v in df.iloc[:30, j])
                if "開示" in col_text or "検討" in col_text:
                    disc_col = j
                    break
            for i in range(len(df)):
                t = _canon_code(df.iat[i, best_col])
                if t is None or t not in universe:
                    continue
                if disc_col is not None:
                    val = _norm(df.iat[i, disc_col])
                    status[t] = ("開示" in val and "未" not in val) or "済" in val
                else:
                    status[t] = True  # presence on the list = disclosed
        overlap = len(status)
        print(f"    catalysts: reform list from {url.rsplit('/',1)[-1]} "
              f"covers {overlap}/{len(universe)} tickers")
        if overlap >= max(10, len(universe) // 5):
            return status
    return None


# ──────────────────────────────────────────────────────────────────────────
# State accumulation & public entry point
# ──────────────────────────────────────────────────────────────────────────
def update_catalysts(state_path: str | Path,
                     universe_meta: list[dict]) -> dict:
    state_path = Path(state_path)
    state = {"events": {}, "processedEdinet": [], "processedTdnet": [],
             "edinetMap": {}, "reform": {}, "reformFetchedAt": None}
    if state_path.exists():
        try:
            state.update(json.loads(state_path.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"    ! catalyst_state unreadable, rebuilding: {e}",
                  file=sys.stderr)

    universe = {m["ticker"] for m in universe_meta}
    meta = {"edinet": False, "tdnet": False, "reform": False}
    today = _jst_today()
    recent_days = [(today - timedelta(days=i)) for i in range(BACKFILL_DAYS)]

    def add_event(ticker: str, ev: dict):
        arr = state["events"].setdefault(ticker, [])
        key = (ev["date"], ev["type"], ev.get("filer"), ev["title"][:40])
        if any((e["date"], e["type"], e.get("filer"), e["title"][:40]) == key
               for e in arr):
            return
        arr.append(ev)

    # ── EDINET ──────────────────────────────────────────────────────────
    api_key = os.environ.get("EDINET_API_KEY")
    if api_key:
        emap = _load_edinet_map(state, universe)
        if emap:
            done = set(state.get("processedEdinet", []))
            n_ev = 0
            for d in recent_days:
                ymd_dash = d.strftime("%Y-%m-%d")
                if ymd_dash in done or d.date() >= today.date():
                    continue  # skip today (list still filling) & processed
                for ticker, ev in _scan_edinet_day(ymd_dash, api_key, emap):
                    add_event(ticker, ev)
                    n_ev += 1
                done.add(ymd_dash)
                time.sleep(0.3)
            state["processedEdinet"] = sorted(done)[-30:]
            meta["edinet"] = True
            print(f"    catalysts: edinet scan done (+{n_ev} 5% events)")
    else:
        print("    catalysts: EDINET_API_KEY not set — skipping 5% reports")

    # ── TDnet ───────────────────────────────────────────────────────────
    try:
        done = set(state.get("processedTdnet", []))
        n_ev = 0
        for d in recent_days:
            ymd = d.strftime("%Y%m%d")
            if ymd in done:
                continue
            events = _scan_tdnet_day(ymd)
            for code, ev in events:
                if code in universe:
                    add_event(code, ev)
                    n_ev += 1
            # only mark past days processed; rescan today on later runs
            if d.date() < today.date():
                done.add(ymd)
        state["processedTdnet"] = sorted(done)[-30:]
        meta["tdnet"] = True
        print(f"    catalysts: tdnet scan done (+{n_ev} keyword events)")
    except Exception as e:
        print(f"    ! tdnet scan failed: {e}", file=sys.stderr)

    # ── JPX reform list ─────────────────────────────────────────────────
    # NOTE: JPX publishes capital-efficiency disclosures only as per-company
    # PDFs (no machine-readable issue list), so per-stock reform tagging is not
    # available from a free structured source. Left out by design; the
    # fetch_reform_status() helper is retained for future use.
    meta["reform"] = False

    # ── prune & persist ─────────────────────────────────────────────────
    cutoff = (today - timedelta(days=EVENT_WINDOW_DAYS)).strftime("%Y-%m-%d")
    for t in list(state["events"].keys()):
        arr = [e for e in state["events"][t] if e["date"] >= cutoff]
        arr.sort(key=lambda e: e["date"], reverse=True)
        state["events"][t] = arr[:MAX_EVENTS_PER_STOCK]
        if not arr:
            del state["events"][t]

    try:
        state_path.write_text(json.dumps(state, ensure_ascii=False),
                              encoding="utf-8")
    except Exception as e:
        print(f"    ! catalyst_state write failed: {e}", file=sys.stderr)

    n_stocks = len(state["events"])
    n_total = sum(len(v) for v in state["events"].values())
    print(f"    catalysts: {n_total} live events across {n_stocks} stocks; "
          f"reform tags: {len(state.get('reform', {}))}")
    return {"events": state["events"], "reform": state.get("reform", {}),
            "meta": meta}
