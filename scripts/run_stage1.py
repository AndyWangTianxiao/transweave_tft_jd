#!/usr/bin/env python3
"""
Stage 1: Full data pipeline — OHLCV + onchain → processed.

Flow:
  1. Fetch OHLCV via ccxt (always).
  2. Onchain: if data/processed/onchain/{ASSET}_onchain_hourly.parquet exist for all assets,
     skip Dune fetch and onchain processing (use cached). Otherwise fetch from Dune + process.
  3. Preprocess OHLCV (1min → 15min etc.).

Usage:
  python scripts/run_stage1.py
  python scripts/run_stage1.py --config config.yaml
  python scripts/run_stage1.py --force-onchain   # Always fetch from Dune (ignore cached onchain)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _onchain_complete(config_path: str, assets: list[str]) -> bool:
    """Return True if processed onchain parquet exists for all assets."""
    import yaml
    cfg = yaml.safe_load((ROOT / config_path).read_text())
    out_dir = Path(cfg["paths"]["processed_onchain"])
    for asset in assets:
        p = out_dir / f"{asset}_onchain_hourly.parquet"
        if not p.exists():
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1: OHLCV fetch + onchain + preprocess")
    ap.add_argument("--config", default="config.yaml", help="Config path")
    ap.add_argument(
        "--force-onchain",
        action="store_true",
        help="Always fetch onchain from Dune (ignore cached processed/onchain)",
    )
    args = ap.parse_args()

    import yaml
    config = yaml.safe_load((ROOT / args.config).read_text())
    assets = list(config["stage1"]["assets"].keys())

    print("=" * 60)
    print("Stage 1: Data pipeline (OHLCV + onchain → processed)")
    print("=" * 60)

    # 1. OHLCV fetch (always)
    print("\n[1/3] Fetching OHLCV via ccxt...")
    from src.data.fetcher import run_fetch
    run_fetch(config_path=args.config, assets=assets)
    print("[OK] OHLCV fetch done.")

    # 2. Onchain: Dune + process, or skip if cached
    use_cached = _onchain_complete(args.config, assets) and not args.force_onchain
    if use_cached:
        print("\n[2/3] Onchain: using cached data/processed/onchain/*.parquet (skip Dune)")
        print("      To force Dune fetch: --force-onchain")
    else:
        print("\n[2/3] Onchain: fetch from Dune + process...")
        try:
            from src.data.dune_fetcher import run_fetch as run_dune
            run_dune(config_path=args.config)
        except RuntimeError as e:
            if "DUNE_API_KEY" in str(e):
                print(
                    "[WARN] Dune fetch skipped (DUNE_API_KEY not set). "
                    "Using cached onchain if present."
                )
                if not _onchain_complete(args.config, assets):
                    raise RuntimeError(
                        "No cached onchain data. Set DUNE_API_KEY and rerun, "
                        "or clone repo with data/processed/onchain/ included."
                    ) from e
            else:
                raise
        from src.data.onchain import run_onchain
        run_onchain(config_path=args.config, assets=assets)
    print("[OK] Onchain ready.")

    # 3. Preprocess OHLCV
    print("\n[3/3] Preprocessing OHLCV (1min → 15min, log return, jump)...")
    from src.data.preprocessor import run_preprocess
    run_preprocess(config_path=args.config, assets=assets)
    print("[OK] Preprocess done.")

    print("\n" + "=" * 60)
    print("Stage 1 complete. Next: bash scripts/run_full_pipeline.sh")
    print("=" * 60)


if __name__ == "__main__":
    main()
