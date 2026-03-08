"""
Stage 5: Regime identification and transfer metrics.
Per doc/stage5_regime.md: paper formula (12) 3-term, W_crit = 2√(σ̄²+λ̄), Phase 2 driven.
"""

import json
import sys
from pathlib import Path

import numpy as np


def _to_json_serializable(obj):
    """Convert numpy types to native Python for json.dump."""
    if isinstance(obj, (np.bool_, np.integer)):
        return bool(obj) if isinstance(obj, np.bool_) else int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    return obj

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.jump_diffusion import (
    fit_rolling_mle,
    load_asset_train_returns,
)
from src.transfer.regime import fit_regime_all_assets, load_regime_result
from src.transfer.metrics import (
    compute_w_jd,
    compute_rho_regime,
    compute_s_transfer,
    compute_rolling_w_jd,
    compute_w_crit,
    spectral_transfer_condition,
    time_change_diagnostic_from_sequences,
    make_transfer_decision,
    regime_diagnostics,
)
from src.transfer.feature_shift import compute_feature_shift, compute_s_transfer_adjusted
from src.behavioral.weakness import (
    compute_rolling_delta_wpt,
    compute_delta_wpt_ks,
)


def main(config_path: str = "config.yaml"):
    root = Path(__file__).resolve().parents[1]
    import yaml
    with open(root / config_path) as f:
        config = yaml.safe_load(f)
    ckpt_dir = root / config["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    assets = ["ETH", "BTC", "SOL", "DOGE"]
    dt = 1.0 / config["training"]["bars_per_year"]
    xfer = config.get("transfer", {})
    spectral_n = xfer.get("spectral_n", 96288)
    spectral_delta = xfer.get("spectral_delta", 0.05)
    ratio_threshold = xfer.get("time_transform_ratio_threshold", 0.5)
    rolling_window = xfer.get("rolling_window_bars", 2880)
    rolling_stride = xfer.get("rolling_stride_bars", 2880)

    # Task 5.0: Rolling MLE for all assets (v3: 30-day window)
    theta_seqs = {}
    for asset in assets:
        try:
            r = load_asset_train_returns(asset, config_path)
            seq = fit_rolling_mle(r, rolling_window, rolling_stride, config_path)
            if seq:
                theta_seqs[asset] = seq
        except FileNotFoundError as e:
            print(f"Skipping {asset}: {e}")

    if "ETH" not in theta_seqs or not theta_seqs["ETH"]:
        raise RuntimeError("ETH rolling MLE failed; cannot run Stage 5")
    print(f"Rolling MLE (30d): n_windows={len(theta_seqs['ETH'])}")

    # Task 5.1: Regime (M4 full PCA)
    fit_regime_all_assets(assets, config_path)

    regime_eth = load_regime_result("ETH", config_path)
    V_eth = regime_eth["eigenvectors"]
    Lambda_eth = regime_eth["eigenvalues"]

    report = {"pairs": [], "summary": {}, "version": "v3_paper_formula"}
    for target in ["BTC", "SOL", "DOGE"]:
        try:
            regime_t = load_regime_result(target, config_path)
        except FileNotFoundError as e:
            print(f"Skipping ETH->{target}: {e}")
            continue
        if target not in theta_seqs or not theta_seqs[target]:
            print(f"Skipping ETH->{target}: target rolling MLE failed")
            continue

        theta_seq_eth = theta_seqs["ETH"]
        theta_seq_t = theta_seqs[target]

        # v3: W_JD paper formula (12), W_crit = 2√(σ̄²+λ̄), no burstiness
        rolling_diag = compute_rolling_w_jd(theta_seq_eth, theta_seq_t)
        w_jd_effective = rolling_diag["w_jd_effective"]
        w_crit = compute_w_crit(theta_seq_eth, theta_seq_t)
        w_jd_semantic = compute_w_jd(
            theta_seq_eth[0], theta_seq_t[0], method="semantic"
        )

        # v2: ρ 排除平稳，无 null 校正
        rho_result = compute_rho_regime(
            V_eth,
            regime_t["eigenvectors"],
            eigenvalues_a=Lambda_eth,
            eigenvalues_b=regime_t["eigenvalues"],
            config_path=config_path,
        )
        rho = rho_result["rho"]

        # M3: ΔW_PT p90 + KS
        wpt_result = compute_rolling_delta_wpt(theta_seq_eth, theta_seq_t, dt, config_path)
        delta_w_pt = wpt_result["delta_wpt"]
        ks_result = compute_delta_wpt_ks(wpt_result["wpt_a_list"], wpt_result["wpt_b_list"])

        # S_transfer (v2: use rho)
        s_result = compute_s_transfer(
            w_jd_effective, rho, delta_w_pt,
            w_crit=w_crit,
        )
        s_transfer = s_result["s_transfer"]

        # M6: Time change diagnostic
        tc_result = time_change_diagnostic_from_sequences(
            theta_seq_eth, theta_seq_t, ratio_threshold
        )

        spectral = spectral_transfer_condition(
            Lambda_eth, regime_t["eigenvalues"], rho, spectral_n, spectral_delta
        )

        try:
            fshift = compute_feature_shift("ETH", target, config_path=config_path)
            s_adjusted = compute_s_transfer_adjusted(
                s_transfer, fshift["D_H"], fshift["missing_penalty"]
            )
        except Exception:
            fshift = {"D_H": None, "missing_penalty": 1.0}
            s_adjusted = s_transfer

        decision_result = make_transfer_decision(
            w_jd_effective, w_crit, rho, delta_w_pt,
            s_transfer=s_transfer,
            s_transfer_adjusted=s_adjusted,
            w_jd_effective=w_jd_effective,
            config_path=config_path,
        )
        decision = decision_result["decision"] if isinstance(decision_result, dict) else decision_result

        try:
            means_eth = regime_eth.get("means", np.zeros((regime_eth["n_states"], 2)))
            means_t = regime_t.get("means", np.zeros((regime_t["n_states"], 2)))
            reg_diag = regime_diagnostics(
                regime_eth["transition_matrix"],
                regime_t["transition_matrix"],
                means_eth,
                means_t,
            )
        except Exception:
            reg_diag = {}

        pair_report = {
            "source": "ETH",
            "target": target,
            "w_jd_effective": w_jd_effective,
            "w_jd_semantic": w_jd_semantic,
            "w_jd_p90": rolling_diag.get("w_jd_p90", rolling_diag["w_jd_mean"]),
            "w_jd_mean": rolling_diag["w_jd_mean"],
            "w_crit": w_crit,
            "w_jd_over_w_crit": w_jd_effective / (w_crit + 1e-10),
            "rho": rho,
            "delta_wpt_p90": delta_w_pt,
            "delta_wpt_mean": wpt_result["delta_wpt_mean"],
            "delta_wpt_ks": ks_result["ks_stat"],
            "delta_wpt_ks_pvalue": ks_result["ks_pvalue"],
            "s_transfer": s_transfer,
            "decision": decision,
            "decision_reason": decision_result.get("reason", "") if isinstance(decision_result, dict) else "",
            "time_change": tc_result,
            "spectral": spectral,
            "rolling": _to_json_serializable(rolling_diag),
            "w_pt": _to_json_serializable({
                "delta_per_window": wpt_result.get("delta_per_window", []),
                "n_windows": wpt_result["n_windows"],
            }),
            "feature_shift": _to_json_serializable(
                {k: v for k, v in fshift.items() if k != "psi_per_feature"}
            ) if isinstance(fshift.get("D_H"), (int, float)) else {},
            "s_transfer_adjusted": float(s_adjusted) if isinstance(s_adjusted, (int, float)) else s_transfer,
            "regime_diagnostics": _to_json_serializable(
                {k: v for k, v in reg_diag.items() if k not in ("pi_a", "pi_b")}
            ),
        }
        report["pairs"].append(pair_report)
        print(f"ETH->{target}: W_JD={w_jd_effective:.3f}, W_crit={w_crit:.3f}, "
              f"ρ={rho:.3f}, ΔW_PT={delta_w_pt:.4f}, S={s_transfer:.3f}, decision={decision}")

    report_path = ckpt_dir / "stage5_transfer_report.json"
    report_serializable = _to_json_serializable(report)
    with open(report_path, "w") as f:
        json.dump(report_serializable, f, indent=2)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
