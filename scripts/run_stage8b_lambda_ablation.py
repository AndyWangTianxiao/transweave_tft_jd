"""
Stage 8B Task B1: λ prior ablation. Per doc/stage8b_ablation_explore.md.

Trains:
  TFT-JD:  A (lambda_prior_weight=0.05) | B (lambda_prior_weight=0)
  Static-JD:  A (MAP prior) | B (pure MLE, prior_weight=0)

Outputs: checkpoints + stage8b_lambda_ablation.json (metadata for notebook).
Evaluation: use notebooks/verify_stage8b_lambda.ipynb.

Usage: python scripts/run_stage8b_lambda_ablation.py [--skip-tft] [--skip-static]
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Stage 8B λ prior ablation")
    ap.add_argument("--skip-tft", action="store_true", help="Skip TFT-JD training")
    ap.add_argument("--skip-static", action="store_true", help="Skip Static-JD fitting")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta = {"tft": {}, "static": {}, "version": "stage8b_v1"}

    # --- TFT-JD: A (prior) vs B (no prior) ---
    if not args.skip_tft:
        train_script = str(ROOT / "scripts" / "train_tft_jd_eth.py")
        for label, weight, suffix in [
            ("A_prior", 0.05, "_lambda_prior"),
            ("B_no_prior", 0.0, "_lambda0"),
        ]:
            ckpt_name = f"eth_tft_jd{suffix}.ckpt"
            if (ckpt_dir / ckpt_name).exists():
                print(f"TFT {label} exists: {ckpt_name}, skip")
                meta["tft"][label] = {"ckpt": ckpt_name, "lambda_prior_weight": weight, "skipped": True}
                continue
            print(f"TFT {label}: lambda_prior_weight={weight} -> {ckpt_name}")
            rc = subprocess.run(
                [
                    sys.executable, train_script,
                    "--use_weakness", "false",
                    "--mode", "scratch",
                    "--lambda_prior_weight", str(weight),
                    "--ckpt_suffix", suffix,
                ],
                cwd=str(ROOT),
            )
            if rc.returncode != 0:
                print(f"TFT {label} failed (exit {rc.returncode})")
                sys.exit(rc.returncode)
            meta["tft"][label] = {"ckpt": ckpt_name, "lambda_prior_weight": weight}
    else:
        meta["tft"]["skipped"] = True

    # --- Static-JD: A (MAP) vs B (MLE) vs B_mle_unconstrained ---
    if not args.skip_static:
        from src.models.jump_diffusion import load_eth_train_returns, fit_static_mle

        r_train = load_eth_train_returns()
        # B_mle_unconstrained: no prior, loose bounds -> degenerate λ (for report fig4)
        for label, prior_w, out_name, lm_override, sj_override in [
            ("A_map", 5000.0, "eth_static_jd_params_map.json", None, None),
            ("B_mle", 0.0, "eth_static_jd_params_mle.json", None, None),
            ("B_mle_unconstrained", 0.0, "eth_static_jd_params_mle_unconstrained.json", 1e6, 1e-6),
        ]:
            if (ckpt_dir / out_name).exists():
                print(f"Static {label} exists: {out_name}, skip")
                meta["static"][label] = {"params_file": out_name, "prior_weight": prior_w, "skipped": True}
                continue
            print(f"Static {label}: prior_weight={prior_w} -> {out_name}")
            fit_static_mle(
                r_train,
                config_path="config.yaml",
                out_filename=out_name,
                prior_weight_override=prior_w,
                lambda_max_override=lm_override,
                sigma_j_min_override=sj_override,
            )
            meta["static"][label] = {"params_file": out_name, "prior_weight": prior_w}
    else:
        meta["static"]["skipped"] = True

    out_path = ckpt_dir / "stage8b_lambda_ablation.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata -> {out_path}")


if __name__ == "__main__":
    main()
