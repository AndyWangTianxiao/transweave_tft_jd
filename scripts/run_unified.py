"""
Stage 7: Unified framework. Per doc/stage7_unify.md.
Orchestrates Stage 2 -> 3 -> 5 -> 6 with --mode skip/force and --stage.
Aligns with Stage 6 outputs: transfer_map_{target}.pt, weak_model_{target}.pt.

Usage:
  python scripts/run_unified.py                    # default: skip cache
  python scripts/run_unified.py --mode force       # always run, overwrite
  python scripts/run_unified.py --stage 5,6        # run only stages 5 and 6
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.unified.framework import run_algorithm_1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 7: Unified Algorithm 1 pipeline (Stage 2 -> 3 -> 5 -> 6)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="skip",
        choices=["skip", "force"],
        help="skip: use cache if exists; force: always run, overwrite",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Stages to run: all, or comma-separated e.g. 5,6",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config file path",
    )
    args = parser.parse_args()

    stages: list[int]
    if args.stage.lower() == "all":
        stages = [2, 3, 5, 6]
    else:
        stages = [int(s.strip()) for s in args.stage.split(",") if s.strip()]

    print(f"[run_unified] mode={args.mode}, stages={stages}")
    summary = run_algorithm_1(
        mode=args.mode,
        stages=stages,
        config_path=args.config,
    )
    print("\n[run_unified] Summary:")
    for k, v in summary.get("stage_status", {}).items():
        print(f"  Stage {k}: {v}")
    if "diagnostics_path" in summary:
        print(f"  Diagnostics: {summary['diagnostics_path']}")


if __name__ == "__main__":
    main()
