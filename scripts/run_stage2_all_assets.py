#!/usr/bin/env python3
"""
Stage 2 for all assets: empirical jump estimation -> Static JD MAP.

For each of ETH, BTC, SOL, DOGE:
  1. Run BNS + Lee-Mykland empirical estimation -> {asset}_empirical_jump_params.json
  2. Run Static MAP with p_center from empirical -> {asset}_static_jd_params.json

Usage:
  python scripts/run_stage2_all_assets.py
  python scripts/run_stage2_all_assets.py --assets ETH,BTC
"""
import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSETS = ["ETH", "BTC", "SOL", "DOGE"]


def _run_empirical(asset: str) -> None:
    """Run empirical jump estimation for one asset (no torch dependency)."""
    spec = importlib.util.spec_from_file_location(
        "empirical_jump", ROOT / "src" / "evaluation" / "empirical_jump.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_and_save(asset=asset)


def _run_static_jd(asset: str) -> None:
    """Run Static MAP for one asset using its empirical p_center."""
    from src.models.jump_diffusion import load_asset_train_returns, fit_static_mle

    config_path = str(ROOT / "config.yaml")
    r_train = load_asset_train_returns(asset, config_path)
    out_filename = f"{asset.lower()}_static_jd_params.json"
    fit_static_mle(
        r_train,
        config_path=config_path,
        out_filename=out_filename,
        asset=asset,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2: empirical + static JD for all assets")
    ap.add_argument(
        "--assets", type=str, default=",".join(ASSETS),
        help=f"Comma-separated assets (default: {','.join(ASSETS)})",
    )
    args = ap.parse_args()
    assets = [a.strip().upper() for a in args.assets.split(",")]

    for asset in assets:
        print(f"\n{'='*60}")
        print(f"  {asset}: Empirical jump estimation")
        print(f"{'='*60}")
        _run_empirical(asset)

        print(f"\n{'='*60}")
        print(f"  {asset}: Static MAP (p_center from empirical)")
        print(f"{'='*60}")
        _run_static_jd(asset)

    print(f"\n{'='*60}")
    print(f"  All done: {', '.join(assets)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
