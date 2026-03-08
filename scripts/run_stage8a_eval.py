"""
Stage 8a Task 3: Run evaluations. Per doc/stage8a_audit_present.md.
Outputs: stage8a_eth_ablation.json, stage8a_transfer_eval.json, stage8a_s_correlation.json,
         stage8a_target_tft.json, stage8a_transweave_vs_target_tft.json.
If target TFT checkpoints (base, scratch+W, finetune+W per asset) are missing, auto-runs
run_train_target_tft.py (6+3 parallel). Use --skip-target-train to skip.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tft_dataset import load_returns
from src.evaluation import (
    evaluate_gbm_baseline,
    nll_per_sample,
    pit_uniformity_test,
    compute_pit,
    s_transfer_vs_nll,
    theorem_5_1_verification,
    var_backtest,
)
from src.models import jump_diffusion as jd
from src.models import losses


def _to_json(obj):
    """Convert numpy types for JSON."""
    if isinstance(obj, (np.bool_, np.integer)):
        return bool(obj) if isinstance(obj, np.bool_) else int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(x) for x in obj]
    return obj


def run_eth_ablation(config_path: str = "config.yaml") -> dict:
    """ETH ablation: GBM, Static-JD, TFT-JD, TFT-JD+W. Per Task 3.1."""
    cfg = yaml.safe_load(open(ROOT / config_path))
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    dt = 1.0 / cfg["training"]["bars_per_year"]
    n_max = cfg["training"].get("jd_truncation_n", 10)

    r_train = load_returns("ETH", "train", config_path)
    r_test = load_returns("ETH", "test", config_path)
    r_train = np.asarray(r_train, dtype=np.float64)
    r_test = np.asarray(r_test, dtype=np.float64)

    out = {"models": {}, "dt": dt, "n_train": len(r_train), "n_test": len(r_test)}

    # GBM
    gbm = evaluate_gbm_baseline(r_train, r_test, dt)
    out["models"]["GBM"] = gbm

    # Static-JD
    static_path = ckpt_dir / "eth_static_jd_params.json"
    if static_path.exists():
        with open(static_path) as f:
            params = json.load(f)
        mu = params["mu"]
        sigma = params["sigma"]
        lam = params["lambda"]
        mu_J = params["mu_J"]
        sigma_J = params["sigma_J"]
        r_t = torch.tensor(r_test, dtype=torch.float64)
        nll = -jd.log_density(r_t, mu, sigma, lam, mu_J, sigma_J, dt, n_max).mean().item()
        crps = losses.crps_mc(r_t, mu, sigma, lam, mu_J, sigma_J, dt, 500, 42).item()
        theta_static = np.array([[mu, sigma, lam, mu_J, sigma_J]] * len(r_test))
        var_res = var_backtest(r_test, theta_static, alpha=0.05, dt=dt, n_max=n_max)
        pit_vals = compute_pit(r_test, theta_static, dt, n_max)
        pit_test = pit_uniformity_test(pit_vals)
        out["models"]["Static-JD"] = {
            "nll": nll,
            "crps": crps,
            "var_breach_rate": var_res["breach_rate"],
            "var_kupiec_pvalue": var_res["kupiec_pvalue"],
            "pit_ks_stat": pit_test["ks_stat"],
            "pit_ks_pvalue": pit_test["p_value"],
        }
    else:
        out["models"]["Static-JD"] = {"error": "eth_static_jd_params.json not found"}

    # TFT-JD, TFT-JD+W: require checkpoint
    for name, ckpt_name in [("TFT-JD", "eth_tft_jd.ckpt"), ("TFT-JD+W", "eth_tft_jd_w_finetune.ckpt")]:
        ckpt_path = ckpt_dir / ckpt_name
        if not ckpt_path.exists():
            out["models"][name] = {"error": f"{ckpt_name} not found"}
            continue
        try:
            from src.models.tft_jd import build_tft_jd, infer_eth_theta
            from src.data.tft_dataset import load_feature_arrays, get_split_indices

            model = build_tft_jd()
            ck = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ck["model_state_dict"], strict=True)
            X_hist, Z_future, y, split_arr = load_feature_arrays("ETH", config_path)
            idx = get_split_indices(split_arr)
            test_mask = np.asarray(split_arr) == "test"
            X_test = X_hist[test_mask]
            Z_test = Z_future[test_mask]
            r_test_tft = y[test_mask]
            theta_list = infer_eth_theta(model, X_test, Z_test, batch_size=2048, device="cpu")
            theta_test = np.array([
                [t["mu"], t["sigma"], t["lam"], t["mu_j"], t["sigma_j"]]
                for t in theta_list
            ], dtype=np.float64)
            r_t = torch.tensor(r_test_tft, dtype=torch.float64)
            nll = -jd.log_density(
                r_t,
                torch.tensor(theta_test[:, 0]),
                torch.tensor(theta_test[:, 1]),
                torch.tensor(theta_test[:, 2]),
                torch.tensor(theta_test[:, 3]),
                torch.tensor(theta_test[:, 4]),
                dt, n_max,
            ).mean().item()
            # crps_mc expects torch tensors for per-step params (uses .numel())
            theta_t = torch.tensor(theta_test, dtype=torch.float64)
            crps = losses.crps_mc(
                r_t,
                theta_t[:, 0], theta_t[:, 1], theta_t[:, 2],
                theta_t[:, 3], theta_t[:, 4],
                dt, 500, 42,
            ).item()
            var_res = var_backtest(r_test_tft, theta_test, alpha=0.05, dt=dt, n_max=n_max)
            pit_vals = compute_pit(r_test_tft, theta_test, dt, n_max)
            pit_test = pit_uniformity_test(pit_vals)
            out["models"][name] = {
                "nll": nll,
                "crps": crps,
                "var_breach_rate": var_res["breach_rate"],
                "var_kupiec_pvalue": var_res["kupiec_pvalue"],
                "pit_ks_stat": pit_test["ks_stat"],
                "pit_ks_pvalue": pit_test["p_value"],
            }
        except Exception as e:
            out["models"][name] = {"error": str(e)}

    return out


def ensure_target_tft_checkpoints(config_path: str, skip: bool = False, sequential: bool = False) -> list:
    """
    Check for 3 TFT variants per asset (base, scratch+W, finetune+W). If any missing and not skip,
    run run_train_target_tft.py (Phase 1: 6 parallel, Phase 2: 3 parallel).
    Returns list of assets that had training (for logging).
    """
    cfg = yaml.safe_load(open(ROOT / config_path))
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    required = [
        f"{a}_tft_jd.ckpt" for a in ["BTC", "SOL", "DOGE"]
    ] + [
        f"{a}_tft_jd_w_scratch.ckpt" for a in ["BTC", "SOL", "DOGE"]
    ] + [
        f"{a}_tft_jd_w_finetune.ckpt" for a in ["BTC", "SOL", "DOGE"]
    ]
    missing = [n for n in required if not (ckpt_dir / n).exists()]
    if skip:
        for n in missing:
            print(f"   [{n}] not found (--skip-target-train)")
        return []
    if not missing:
        return []

    print(f"   Missing {len(missing)} target TFT checkpoints, running run_train_target_tft.py...")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_train_target_tft.py")]
    if sequential:
        cmd.append("--sequential")
    ret = subprocess.run(cmd, cwd=str(ROOT))
    if ret.returncode != 0:
        print(f"   run_train_target_tft.py failed (exit {ret.returncode})")
        return []
    return ["BTC", "SOL", "DOGE"]


def _eval_one_ckpt(ckpt_path: Path, asset: str, config_path: str, dt: float, n_max: int) -> dict:
    """Evaluate one TFT checkpoint. Returns {nll, crps, n_samples} or {error}."""
    from src.models.tft_jd import build_tft_jd, infer_eth_theta
    from src.data.tft_dataset import load_feature_arrays
    try:
        model = build_tft_jd()
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model_state_dict"], strict=True)
        X_hist, Z_future, y, split_arr = load_feature_arrays(asset, config_path)
        test_mask = np.asarray(split_arr) == "test"
        X_test, Z_test, r_test = X_hist[test_mask], Z_future[test_mask], y[test_mask]
        theta_list = infer_eth_theta(model, X_test, Z_test, batch_size=2048, device="cpu")
        theta_test = np.array(
            [[t["mu"], t["sigma"], t["lam"], t["mu_j"], t["sigma_j"]] for t in theta_list
        ], dtype=np.float64)
        r_t = torch.tensor(r_test, dtype=torch.float64)
        theta_t = torch.tensor(theta_test, dtype=torch.float64)
        nll = -jd.log_density(
            r_t, theta_t[:, 0], theta_t[:, 1], theta_t[:, 2],
            theta_t[:, 3], theta_t[:, 4], dt, n_max,
        ).mean().item()
        crps = losses.crps_mc(
            r_t, theta_t[:, 0], theta_t[:, 1], theta_t[:, 2],
            theta_t[:, 3], theta_t[:, 4], dt, 500, 42,
        ).item()
        return {"nll": nll, "crps": crps, "n_samples": len(r_test)}
    except Exception as e:
        return {"error": str(e)}


def run_target_tft_eval(config_path: str = "config.yaml", auto_train: bool = True, sequential_train: bool = False) -> dict:
    """
    Evaluate target TFT (base, scratch+W, finetune+W) per asset for comparison with TransWeave.
    If checkpoints missing and auto_train=True, runs run_train_target_tft.py first.
    """
    if auto_train:
        ensure_target_tft_checkpoints(config_path, skip=False, sequential=sequential_train)
    else:
        ensure_target_tft_checkpoints(config_path, skip=True)

    cfg = yaml.safe_load(open(ROOT / config_path))
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    dt = 1.0 / cfg["training"]["bars_per_year"]
    n_max = cfg["training"].get("jd_truncation_n", 10)

    out = {}
    for asset in ["BTC", "SOL", "DOGE"]:
        variants = {}
        for name, ckpt_name in [
            ("base", f"{asset}_tft_jd.ckpt"),
            ("scratch_w", f"{asset}_tft_jd_w_scratch.ckpt"),
            ("finetune_w", f"{asset}_tft_jd_w_finetune.ckpt"),
        ]:
            p = ckpt_dir / ckpt_name
            if p.exists():
                variants[name] = _eval_one_ckpt(p, asset, config_path, dt, n_max)
            else:
                variants[name] = {"error": f"{ckpt_name} not found"}
        out[asset] = variants
        # Backward compat: flat nll/crps from finetune_w (or best available)
        if "finetune_w" in variants and "nll" in variants["finetune_w"]:
            out[asset]["nll"] = variants["finetune_w"]["nll"]
            out[asset]["crps"] = variants["finetune_w"]["crps"]
        elif "scratch_w" in variants and "nll" in variants["scratch_w"]:
            out[asset]["nll"] = variants["scratch_w"]["nll"]
            out[asset]["crps"] = variants["scratch_w"]["crps"]
        elif "base" in variants and "nll" in variants["base"]:
            out[asset]["nll"] = variants["base"]["nll"]
            out[asset]["crps"] = variants["base"]["crps"]

    return {"target_tft_jd": out, "dt": dt}


def run_transfer_eval(config_path: str = "config.yaml") -> dict:
    """Transfer evaluation: 3 targets x 3 methods. Per Task 3.2."""
    cfg = yaml.safe_load(open(ROOT / config_path))
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    stage5_path = ckpt_dir / "stage5_transfer_report.json"
    stage6_path = ckpt_dir / "stage6_experiment_results.json"
    if not stage5_path.exists() or not stage6_path.exists():
        return {"error": "stage5 or stage6 results not found"}
    with open(stage5_path) as f:
        stage5 = json.load(f)
    with open(stage6_path) as f:
        stage6 = json.load(f)
    result = {"stage6_summary": stage6, "stage5_pairs": stage5.get("pairs", [])}
    # Merge target TFT-JD from stage8a_target_tft.json if available
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    target_tft_path = ckpt_dir / "stage8a_target_tft.json"
    if target_tft_path.exists():
        with open(target_tft_path) as f:
            target_tft = json.load(f)
        if "target_tft_jd" in target_tft:
            result["target_tft_jd"] = target_tft["target_tft_jd"]
    return result


def run_s_correlation(config_path: str = "config.yaml") -> dict:
    """S_transfer vs NLL improvement correlation. Per Task 3.3."""
    cfg = yaml.safe_load(open(ROOT / config_path))
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    stage5_path = ckpt_dir / "stage5_transfer_report.json"
    stage6_path = ckpt_dir / "stage6_experiment_results.json"
    if not stage5_path.exists() or not stage6_path.exists():
        return {"error": "stage5 or stage6 results not found"}
    with open(stage5_path) as f:
        stage5 = json.load(f)
    with open(stage6_path) as f:
        stage6 = json.load(f)
    return s_transfer_vs_nll(stage5, stage6)


def run_theorem_verification(config_path: str = "config.yaml") -> dict:
    """Theorem 5.1 verification."""
    cfg = yaml.safe_load(open(ROOT / config_path))
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    stage5_path = ckpt_dir / "stage5_transfer_report.json"
    stage6_path = ckpt_dir / "stage6_experiment_results.json"
    if not stage5_path.exists() or not stage6_path.exists():
        return {"error": "stage5 or stage6 results not found"}
    with open(stage5_path) as f:
        stage5 = json.load(f)
    with open(stage6_path) as f:
        stage6 = json.load(f)
    return theorem_5_1_verification(stage5, stage6)


# TransWeave vs Target-TFT comparison grades (ΔNLL = TransWeave_NLL - Target_TFT_NLL)
# NLL lower is better; ΔNLL ≤ 0 means TransWeave matches or beats Target-TFT.
TRANSFER_VS_TFT_GRADES = [
    (0.0, "excellent"),   # ΔNLL ≤ 0: TransWeave ≥ Target-TFT
    (0.05, "good"),      # 0 < ΔNLL ≤ 0.05: ~95% likelihood ratio
    (0.1, "acceptable"), # 0.05 < ΔNLL ≤ 0.1: ~90% likelihood ratio
    (0.2, "marginal"),   # 0.1 < ΔNLL ≤ 0.2
    (999.0, "poor"),     # ΔNLL > 0.2
]


def _grade_dnll(dnll: float) -> str:
    """Map ΔNLL to grade. Per doc/stage8a_audit_present.md §3.6."""
    for thresh, grade in TRANSFER_VS_TFT_GRADES:
        if dnll <= thresh:
            return grade
    return "poor"


def run_transweave_vs_target_tft(config_path: str = "config.yaml") -> dict:
    """
    Compare TransWeave NLL vs Target-TFT NLL per target. ΔNLL = TransWeave - Target_TFT.
    Grade: excellent (Δ≤0) | good (≤0.05) | acceptable (≤0.1) | marginal (≤0.2) | poor.
    """
    cfg = yaml.safe_load(open(ROOT / config_path))
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    stage6_path = ckpt_dir / "stage6_experiment_results.json"
    target_tft_path = ckpt_dir / "stage8a_target_tft.json"
    if not stage6_path.exists():
        return {"error": "stage6_experiment_results.json not found"}
    if not target_tft_path.exists():
        return {"error": "stage8a_target_tft.json not found (run target TFT eval first)"}
    with open(stage6_path) as f:
        stage6 = json.load(f)
    with open(target_tft_path) as f:
        target_tft = json.load(f)
    tft_jd = target_tft.get("target_tft_jd") or {}
    grade_spec = [
        {"max_delta_nll": t, "grade": g} for t, g in TRANSFER_VS_TFT_GRADES
    ]
    out = {"by_target": {}, "grade_spec": grade_spec}
    for rec in stage6:
        if not isinstance(rec, dict):
            continue
        asset = rec.get("target_asset")
        tw = rec.get("transweave")
        if not asset or not tw or "nll" not in tw:
            continue
        tft_v = tft_jd.get(asset)
        if not tft_v or "nll" not in tft_v:
            out["by_target"][asset] = {"error": f"Target TFT NLL missing for {asset}"}
            continue
        tw_nll = tw["nll"]
        tft_nll = tft_v["nll"]
        dnll = tw_nll - tft_nll
        grade = _grade_dnll(dnll)
        out["by_target"][asset] = {
            "transweave_nll": tw_nll,
            "target_tft_nll": tft_nll,
            "delta_nll": dnll,
            "grade": grade,
        }
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Stage 8a evaluation")
    ap.add_argument("--skip-target-train", action="store_true", help="Skip auto-training missing target TFT checkpoints")
    ap.add_argument("--sequential-target-train", action="store_true", help="Run run_train_target_tft.py with --sequential")
    args = ap.parse_args()

    config_path = "config.yaml"
    ckpt_dir = ROOT / "experiments" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Stage 8a evaluation...")
    print("1. ETH ablation")
    eth_ablation = run_eth_ablation(config_path)
    with open(ckpt_dir / "stage8a_eth_ablation.json", "w") as f:
        json.dump(_to_json(eth_ablation), f, indent=2)
    print(f"   -> stage8a_eth_ablation.json")

    print("2. Target TFT-JD+W (for TransWeave comparison)")
    target_tft = run_target_tft_eval(
        config_path,
        auto_train=not args.skip_target_train,
        sequential_train=args.sequential_target_train,
    )
    with open(ckpt_dir / "stage8a_target_tft.json", "w") as f:
        json.dump(_to_json(target_tft), f, indent=2)
    print(f"   -> stage8a_target_tft.json")
    if "target_tft_jd" in target_tft:
        for a, v in target_tft["target_tft_jd"].items():
            if "nll" in v:
                extra = []
                for k in ["base", "scratch_w", "finetune_w"]:
                    if k in v and isinstance(v[k], dict) and "nll" in v[k]:
                        extra.append(f"{k}={v[k]['nll']:.4f}")
                suf = " (" + ", ".join(extra) + ")" if extra else ""
                print(f"      {a}: NLL={v['nll']:.4f} CRPS={v['crps']:.6f}{suf}")
            else:
                print(f"      {a}: {v.get('error', 'ok')}")

    print("3. Transfer eval (from stage6)")
    transfer_eval = run_transfer_eval(config_path)
    with open(ckpt_dir / "stage8a_transfer_eval.json", "w") as f:
        json.dump(_to_json(transfer_eval), f, indent=2)
    print(f"   -> stage8a_transfer_eval.json")

    print("4. S_transfer vs NLL correlation")
    s_corr = run_s_correlation(config_path)
    with open(ckpt_dir / "stage8a_s_correlation.json", "w") as f:
        json.dump(_to_json(s_corr), f, indent=2)
    print(f"   -> stage8a_s_correlation.json (pearson_r={s_corr.get('pearson_r', 'N/A')})")

    print("5. Theorem 5.1 verification")
    thm = run_theorem_verification(config_path)
    with open(ckpt_dir / "stage8a_theorem_verification.json", "w") as f:
        json.dump(_to_json(thm), f, indent=2)
    print(f"   -> stage8a_theorem_verification.json")

    print("6. TransWeave vs Target-TFT comparison")
    tw_vs_tft = run_transweave_vs_target_tft(config_path)
    with open(ckpt_dir / "stage8a_transweave_vs_target_tft.json", "w") as f:
        json.dump(_to_json(tw_vs_tft), f, indent=2)
    print(f"   -> stage8a_transweave_vs_target_tft.json")
    if "by_target" in tw_vs_tft:
        for a, v in tw_vs_tft["by_target"].items():
            if "grade" in v:
                print(f"      {a}: ΔNLL={v['delta_nll']:.4f} -> {v['grade']}")
            else:
                print(f"      {a}: {v.get('error', 'ok')}")

    print("Done.")


if __name__ == "__main__":
    main()
