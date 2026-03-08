#!/usr/bin/env python3
"""
Estimate empirical jump frequency from ETH data and output recommended p_center.

Run before Stage 2 / TFT training to use data-driven prior:
  python scripts/estimate_jump_prior.py

Output: experiments/checkpoints/eth_empirical_jump_params.json
  - p_center_recommended: use in config static_jd_map.p_center or for TFT lambda_prior_center
  - lambda_recommended: annualized jump intensity
"""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import empirical_jump directly to avoid evaluation package pulling torch
spec = importlib.util.spec_from_file_location(
    "empirical_jump", ROOT / "src" / "evaluation" / "empirical_jump.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Estimate empirical jump frequency")
    ap.add_argument("--asset", default="ETH", choices=["ETH", "BTC", "SOL", "DOGE"],
                     help="Asset to estimate (default: ETH)")
    args = ap.parse_args()
    mod.run_and_save(asset=args.asset)
