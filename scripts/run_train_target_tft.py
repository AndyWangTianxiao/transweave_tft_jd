"""
Train target TFT for BTC/SOL/DOGE, aligned with ETH: base, scratch+W, finetune+W.
Per doc/stage8a_audit_present.md §3.5.

Phase 1: 6 parallel (3 assets × base + scratch+W). Skip if checkpoint exists.
Phase 2: 3 parallel (finetune+W per asset). Requires base to exist.

Usage: python scripts/run_train_target_tft.py [--force] [--sequential]
"""

import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


def _load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _run_train(asset: str, mode: str, use_weakness: str, force: bool) -> int:
    """Run train_tft_jd_target.py. Returns exit code."""
    args = [
        sys.executable,
        str(ROOT / "scripts" / "train_tft_jd_target.py"),
        "--asset", asset,
        "--mode", mode,
        "--use_weakness", use_weakness,
    ]
    if force:
        args.append("--force")
    return subprocess.run(args, cwd=str(ROOT)).returncode


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Train target TFT (base, scratch+W, finetune+W)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing checkpoints")
    ap.add_argument("--sequential", action="store_true", help="Run sequentially instead of parallel")
    args = ap.parse_args()

    cfg = _load_config()
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    assets = ["BTC", "SOL", "DOGE"]

    # Phase 1: base + scratch+W (6 jobs, can run in parallel)
    phase1 = []
    for a in assets:
        if args.force or not (ckpt_dir / f"{a}_tft_jd.ckpt").exists():
            phase1.append((a, "scratch", "false", f"{a}_tft_jd.ckpt"))
        if args.force or not (ckpt_dir / f"{a}_tft_jd_w_scratch.ckpt").exists():
            phase1.append((a, "scratch", "true", f"{a}_tft_jd_w_scratch.ckpt"))

    if phase1:
        print(f"Phase 1: {len(phase1)} jobs (base + scratch+W)")
        if args.sequential:
            for a, mode, uw, name in phase1:
                rc = _run_train(a, mode, uw, args.force)
                print(f"  [{name}] exit={rc}")
                if rc != 0:
                    sys.exit(rc)
        else:
            with ProcessPoolExecutor(max_workers=min(6, len(phase1))) as ex:
                futures = {
                    ex.submit(_run_train, a, mode, uw, args.force): (a, name)
                    for a, mode, uw, name in phase1
                }
                for fut in as_completed(futures):
                    a, name = futures[fut]
                    rc = fut.result()
                    print(f"  [{name}] exit={rc}")
                    if rc != 0:
                        sys.exit(rc)
    else:
        print("Phase 1: all checkpoints exist, skip")

    # Phase 2: finetune+W (3 jobs, requires base)
    phase2 = []
    for a in assets:
        if (ckpt_dir / f"{a}_tft_jd.ckpt").exists():
            if args.force or not (ckpt_dir / f"{a}_tft_jd_w_finetune.ckpt").exists():
                phase2.append((a, "finetune", "true", f"{a}_tft_jd_w_finetune.ckpt"))
        else:
            print(f"  [WARN] {a}_tft_jd.ckpt missing, skip finetune for {a}")

    if phase2:
        print(f"Phase 2: {len(phase2)} jobs (finetune+W)")
        if args.sequential:
            for a, mode, uw, name in phase2:
                rc = _run_train(a, mode, uw, args.force)
                print(f"  [{name}] exit={rc}")
                if rc != 0:
                    sys.exit(rc)
        else:
            with ProcessPoolExecutor(max_workers=min(3, len(phase2))) as ex:
                futures = {
                    ex.submit(_run_train, a, mode, uw, args.force): (a, name)
                    for a, mode, uw, name in phase2
                }
                for fut in as_completed(futures):
                    a, name = futures[fut]
                    rc = fut.result()
                    print(f"  [{name}] exit={rc}")
                    if rc != 0:
                        sys.exit(rc)
    else:
        print("Phase 2: all finetune checkpoints exist, skip")

    print("Done")


if __name__ == "__main__":
    main()
