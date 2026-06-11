"""
유니버스 빌더 — JPX 東証上場銘柄一覧(data_j.xls)에서 시총 상위 N종목을
뽑아 tickers.json을 생성/갱신한다. (로컬에서 실행)

핵심 원칙:
  • 기존 tickers.json의 한국어 이름(nameKo)은 보존 — 새 종목만 영문으로 채움
  • JPX 33업종 → 대시보드 11섹터로 매핑
  • 시총은 yfinance로 조회 (느리므로 1회만 돌리면 됨; 결과가 tickers.json)

실행:
    py scripts/build_universe.py --target 250
    py scripts/build_universe.py --target 250 --dry-run     # 미리보기만

옵션:
    --target N      최종 종목 수 (기본 250)
    --keep-existing 기존 tickers.json의 모든 종목을 무조건 포함 (기본 켜짐)
    --sleep S       yfinance 호출 간 대기 (기본 0.3초)
    --out PATH      출력 경로 (기본 scripts/tickers.json)
    --dry-run       파일 안 쓰고 통계만 출력

회사 SSL 이슈가 있으면:  set UNIVERSE_INSECURE_SSL=1
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd

JPX_LIST_URL = ("https://www.jpx.co.jp/markets/statistics-equities/misc/"
                "tvdivq0000001vg2-att/data_j.xls")
UA = {"User-Agent": "Mozilla/5.0 (nihon-dashboard universe builder)"}

if os.environ.get("UNIVERSE_INSECURE_SSL") == "1":
    _CTX = ssl.create_default_context()
    _CTX.check_hostname = False
    _CTX.verify_mode = ssl.CERT_NONE
else:
    _CTX = None

# JPX「33業種区分」→ dashboard 11-sector vocabulary
SECTOR_MAP = {
    "水産・農林業": "ConsumerStaples",
    "鉱業": "Energy",
    "建設業": "Industrials",
    "食料品": "ConsumerStaples",
    "繊維製品": "ConsumerDiscretionary",
    "パルプ・紙": "Materials",
    "化学": "Materials",
    "医薬品": "Healthcare",
    "石油・石炭製品": "Energy",
    "ゴム製品": "ConsumerDiscretionary",
    "ガラス・土石製品": "Materials",
    "鉄鋼": "Materials",
    "非鉄金属": "Materials",
    "金属製品": "Materials",
    "機械": "Industrials",
    "電気機器": "Technology",
    "輸送用機器": "ConsumerDiscretionary",
    "精密機器": "Technology",
    "その他製品": "Industrials",
    "電気・ガス業": "Utilities",
    "陸運業": "Industrials",
    "海運業": "Industrials",
    "空運業": "Industrials",
    "倉庫・運輸関連業": "Industrials",
    "情報・通信業": "Communication",
    "卸売業": "Industrials",
    "小売業": "ConsumerDiscretionary",
    "銀行業": "Financials",
    "証券、商品先物取引業": "Financials",
    "保険業": "Financials",
    "その他金融業": "Financials",
    "不動産業": "RealEstate",
    "サービス業": "Industrials",
}

SEGMENT_MAP = {
    "プライム（内国株式）": "Prime",
    "スタンダード（内国株式）": "Standard",
    "グロース（内国株式）": "Growth",
    "プライム（外国株式）": "Prime",
    "スタンダード（外国株式）": "Standard",
    "グロース（外国株式）": "Growth",
}


# Nikkei 225 constituent codes (as of 2025-2026; stable large-cap seed used by
# --fast mode so we can build a ~250 universe WITHOUT pulling 1,600 market caps).
NIKKEI225_CODES = [
    "1332","1605","1721","1801","1802","1803","1808","1812","1925","1928","1963",
    "2002","2269","2282","2413","2432","2501","2502","2503","2531","2768","2801",
    "2802","2871","2914","3086","3092","3099","3289","3382","3401","3402","3405",
    "3407","3436","3659","3861","3863","4004","4005","4021","4042","4043","4061",
    "4063","4151","4183","4188","4208","4324","4385","4452","4502","4503","4506",
    "4507","4519","4523","4543","4568","4578","4631","4661","4689","4704","4751",
    "4755","4901","4902","4911","4061","5019","5020","5101","5108","5201","5202",
    "5214","5233","5301","5332","5333","5401","5406","5411","5631","5703","5706",
    "5707","5711","5713","5714","5801","5802","5803","5831","5832","6098","6103",
    "6113","6146","6178","6273","6301","6302","6305","6326","6361","6367","6448",
    "6457","6460","6471","6472","6473","6479","6501","6503","6504","6506","6526",
    "6532","6594","6645","6701","6702","6723","6724","6752","6753","6754","6758",
    "6762","6770","6841","6857","6861","6869","6902","6920","6952","6954","6963",
    "6967","6971","6976","6981","6988","7003","7011","7012","7013","7186","7201",
    "7202","7203","7205","7211","7261","7267","7269","7270","7272","7731","7733",
    "7735","7741","7751","7752","7762","7832","7911","7912","7951","7974","8001",
    "8002","8015","8031","8035","8053","8058","8113","8233","8252","8253","8267",
    "8304","8306","8308","8309","8316","8331","8354","8411","8591","8601","8604",
    "8630","8697","8725","8729","8750","8766","8795","8801","8802","8804","8830",
    "9001","9005","9007","9008","9009","9020","9021","9022","9064","9101","9104",
    "9107","9147","9201","9202","9301","9412","9432","9433","9434","9501","9502",
    "9503","9504","9505","9506","9507","9508","9509","9531","9532","9602","9613",
    "9697","9706","9735","9766","9783","9843","9983","9984",
]


def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
        return r.read()


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {t["ticker"]: t for t in data}
    except Exception as e:
        print(f"  ! could not read existing {path}: {e}", file=sys.stderr)
        return {}


def fetch_jpx_list() -> pd.DataFrame:
    print("  downloading JPX 東証上場銘柄一覧 (data_j.xls)...")
    raw = _get(JPX_LIST_URL)
    df = pd.read_excel(BytesIO(raw), dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"    {len(df)} listed issues, columns: {list(df.columns)}")
    return df


def normalize_jpx(df: pd.DataFrame) -> pd.DataFrame:
    col_code = next(c for c in df.columns if "コード" in c)
    col_name = next(c for c in df.columns if "銘柄名" in c)
    col_seg = next(c for c in df.columns if "市場" in c and "区分" in c)
    col_sec = next((c for c in df.columns if "33業種区分" in c), None)

    out = pd.DataFrame({
        "code": df[col_code].astype(str).str.strip().str.zfill(4),
        "nameJp": df[col_name].astype(str).str.strip(),
        "segRaw": df[col_seg].astype(str).str.strip(),
        "secRaw": (df[col_sec].astype(str).str.strip()
                   if col_sec else ""),
    })
    # keep only 4-digit ordinary listings on the three main segments
    out = out[out["code"].str.match(r"^\d{4}$")]
    out = out[out["segRaw"].isin(SEGMENT_MAP.keys())]
    # drop ETFs/REITs that slip through (segRaw handles most; sector empty → skip later)
    return out.reset_index(drop=True)


def fetch_market_cap(code: str, sleep: float) -> float | None:
    import yfinance as yf
    try:
        info = yf.Ticker(f"{code}.T").fast_info
        mc = info.get("market_cap") if hasattr(info, "get") else getattr(info, "market_cap", None)
        time.sleep(sleep)
        return float(mc) if mc else None
    except Exception:
        time.sleep(sleep)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=250)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "tickers.json"))
    ap.add_argument("--fast", action="store_true",
                    help="Skip market-cap ranking; use Nikkei225 + existing as seed.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    existing = load_existing(out_path)
    print(f"  existing universe: {len(existing)} tickers (nameKo preserved)")

    df = normalize_jpx(fetch_jpx_list())
    print(f"  candidate pool after segment filter: {len(df)}")
    jpx_by_code = {r.code: r for r in df.itertuples()}

    if args.fast:
        # FAST: union of existing + Nikkei225 seed, padded from the prime pool
        # in code order until target. No market-cap calls (instant).
        print("  --fast: Nikkei225 + existing seed (no market-cap ranking)")
        prime_codes = list(df[df["segRaw"].str.startswith("プライム")]["code"])
        chosen_codes = list(existing.keys())
        for c in NIKKEI225_CODES:
            if c not in chosen_codes:
                chosen_codes.append(c)
        for c in prime_codes:
            if len(chosen_codes) >= args.target:
                break
            if c not in chosen_codes:
                chosen_codes.append(c)
        chosen_codes = chosen_codes[:max(args.target, len(existing))]
    else:
        # Rank by market cap — fetch the whole prime pool (slow, ~1,600 calls).
        prime = df[df["segRaw"].str.startswith("プライム")].copy()
        print(f"  prime issues to rank: {len(prime)} (fetching market caps, slow)...")
        caps = []
        for i, row in enumerate(prime.itertuples(), 1):
            caps.append(fetch_market_cap(row.code, args.sleep))
            if i % 100 == 0:
                print(f"    {i}/{len(prime)} caps fetched...")
        prime["mcap"] = caps
        prime = prime.dropna(subset=["mcap"]).sort_values("mcap", ascending=False)
        print(f"  ranked {len(prime)} issues with valid market cap")
        chosen_codes = list(existing.keys())
        for row in prime.itertuples():
            if len(chosen_codes) >= args.target:
                break
            if row.code not in chosen_codes:
                chosen_codes.append(row.code)
        jpx_by_code = {r.code: r for r in prime.itertuples()}

    # Build records (preserve existing nameKo; new issues fall back to JP text)
    records = []
    n_new = 0
    for code in chosen_codes:
        if code in existing:
            records.append(existing[code])
            continue
        r = jpx_by_code.get(code)
        if r is None:
            continue
        sector = SECTOR_MAP.get(getattr(r, "secRaw", ""), "Industrials")
        segment = SEGMENT_MAP.get(getattr(r, "segRaw", ""), "Prime")
        records.append({
            "ticker": code,
            "nameEn": r.nameJp,
            "nameKo": r.nameJp,
            "nameJp": r.nameJp,
            "segment": segment,
            "sector": sector,
        })
        n_new += 1

    # stats
    from collections import Counter
    sec_dist = Counter(r["sector"] for r in records)
    print(f"\n  final universe: {len(records)} tickers "
          f"({len(existing)} kept, {n_new} new)")
    print(f"  sector distribution: {dict(sec_dist)}")

    if args.dry_run:
        print("\n  --dry-run: not writing. Sample new entries:")
        for r in records:
            if r["ticker"] not in existing:
                print(f"    {r['ticker']}  {r['nameJp'][:20]}  {r['sector']}")
        return

    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n  ✓ wrote {len(records)} tickers to {out_path}")


if __name__ == "__main__":
    main()
