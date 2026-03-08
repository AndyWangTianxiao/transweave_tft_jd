"""
Stage 1: Fetch on-chain data from Dune Analytics via API.
Saves incrementally after each query. Supports try mode for low-cost validation.
API key: set DUNE_API_KEY env var (never commit the key).
"""

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config from YAML. Path relative to project root."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def get_dune_queries(config: dict) -> list[dict]:
    """
    Build list of Dune queries from config.
    Each item: {query_id, out_file, desc, merge_key} where merge_key groups SOL Y1/Y2/Y3.
    """
    cfg = config.get("stage1", {}).get("dune", {})
    raw_dir = Path(config["paths"]["raw_onchain"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "_raw").mkdir(exist_ok=True)

    rows = []
    for chain, queries in cfg.get("queries", {}).items():
        for q in queries:
            rows.append({
                "query_id": int(q["query_id"]),
                "out_file": raw_dir / q["out_file"],
                "desc": q.get("desc", ""),
                "merge_key": q.get("merge_key"),  # e.g. "sol_tx_fee" for Y1/Y2/Y3
            })
    return rows


def fetch_one(
    query_id: int,
    out_path: Optional[Path],
    raw_backup_dir: Path,
    client: Any,
) -> bool:
    """
    Fetch one Dune query, save raw backup and optionally CSV.
    If out_path is None (merge group), only save backup.
    """
    try:
        df = client.get_latest_result_dataframe(query_id)
    except Exception as e:
        print(f"  [ERROR] query_id={query_id}: {e}")
        return False

    if df is None or df.empty:
        print(f"  [WARN] query_id={query_id}: empty result")
        return False

    backup = raw_backup_dir / f"query_{query_id}.parquet"
    df.to_parquet(backup, index=False)
    print(f"  [OK] backup -> {backup.name} ({len(df)} rows)")

    if out_path is not None:
        df.to_csv(out_path, index=False)
        print(f"  [OK] save -> {out_path.name}")
    return True


def merge_sol_tx_fee(rows: list[dict], raw_dir: Path, out_path: Path) -> bool:
    """
    Merge SOL Transaction Count Y1/Y2/Y3 into one sol_tx_fee_hourly.csv.
    """
    dfs = []
    for r in rows:
        backup = raw_dir / "_raw" / f"query_{r['query_id']}.parquet"
        if not backup.exists():
            print(f"  [WARN] merge: {backup} not found, skip")
            continue
        dfs.append(pd.read_parquet(backup))

    if not dfs:
        print("  [ERROR] merge: no SOL tx_fee parts found")
        return False

    merged = pd.concat(dfs, ignore_index=True)
    # Dedupe by hour (in case overlap)
    hour_col = "hour" if "hour" in merged.columns else merged.columns[0]
    merged = merged.drop_duplicates(subset=[hour_col], keep="first")
    merged = merged.sort_values(hour_col).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    print(f"  [OK] merged -> {out_path.name} ({len(merged)} rows)")
    return True


def run_fetch(
    try_mode: bool = False,
    config_path: str = "config.yaml",
    api_key: Optional[str] = None,
) -> None:
    """
    Fetch all Dune on-chain data. Saves after each query.
    try_mode: only fetch first 2 queries to verify pipeline (low cost).
    api_key: override env; prefer DUNE_API_KEY env var.
    """
    import os

    api_key = api_key or os.environ.get("DUNE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set DUNE_API_KEY env var, e.g.: export DUNE_API_KEY='your-key'"
        )

    from dune_client.client import DuneClient

    config = load_config(config_path)
    queries = get_dune_queries(config)
    raw_dir = Path(config["paths"]["raw_onchain"])
    raw_backup = raw_dir / "_raw"

    # Try mode: first 2 queries only
    if try_mode:
        queries = queries[:2]
        print("[TRY MODE] Fetching only first 2 queries to verify pipeline.\n")

    client = DuneClient(api_key=api_key)

    # Group SOL tx_fee (Y1/Y2/Y3) for merge
    merge_groups: dict[str, list] = {}
    single_queries = []
    for q in queries:
        if q.get("merge_key"):
            merge_groups.setdefault(q["merge_key"], []).append(q)
        else:
            single_queries.append(q)

    # Fetch single queries (skip if backup exists to save credits)
    for q in single_queries:
        backup_path = raw_backup / f"query_{q['query_id']}.parquet"
        if backup_path.exists():
            print(f"[{q['desc']}] query_id={q['query_id']} (cached)")
            if not q["out_file"].exists():
                pd.read_parquet(backup_path).to_csv(q["out_file"], index=False)
            continue
        print(f"[{q['desc']}] query_id={q['query_id']}")
        fetch_one(q["query_id"], q["out_file"], raw_backup, client)

    # Fetch and merge SOL tx_fee
    for merge_key, group in merge_groups.items():
        out_path = group[0]["out_file"]  # same for all
        print(f"[{merge_key}] fetching {len(group)} parts...")
        for q in group:
            backup_path = raw_backup / f"query_{q['query_id']}.parquet"
            if backup_path.exists():
                print(f"  skip query_id={q['query_id']} (cached)")
                continue
            success = fetch_one(q["query_id"], None, raw_backup, client)
            if not success:
                print(f"  [WARN] query_id={q['query_id']} failed")
        merge_sol_tx_fee(group, raw_dir, out_path)

    print("\nDone.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--try", dest="try_mode", action="store_true", help="Fetch only 2 queries (low cost)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_fetch(try_mode=args.try_mode, config_path=args.config)
