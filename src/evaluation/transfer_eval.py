"""
Transfer evaluation: S_transfer vs NLL improvement, Theorem 5.1 verification.
Per doc/stage8a_audit_present.md Task 2.3.
"""

from typing import Any, Dict, List

import numpy as np
from scipy import stats


def s_transfer_vs_nll(
    stage5_report: dict,
    stage6_results: List[dict],
) -> dict:
    """
    S_transfer vs NLL improvement. Pearson + Spearman correlation.

    Args:
        stage5_report: stage5_transfer_report.json with pairs[].s_transfer, rho, etc.
        stage6_results: stage6_experiment_results.json list of {target_asset, scratch, transweave, ...}
    Returns:
        {"pearson_r": float, "pearson_p": float, "spearman_r": float, "spearman_p": float,
         "pairs": [{"target": str, "s_transfer": float, "delta_nll": float}], "n": int}
    """
    # Build pair lookup from stage5
    s_by_target = {}
    for p in stage5_report.get("pairs", []):
        tgt = p.get("target")
        if tgt:
            s_by_target[tgt] = p.get("s_transfer", np.nan)

    s_list = []
    delta_list = []
    pairs_out = []
    for row in stage6_results:
        tgt = row.get("target_asset")
        if not tgt:
            continue
        scratch = row.get("scratch", {})
        tw = row.get("transweave", {})
        s = s_by_target.get(tgt, np.nan)
        scratch_nll = scratch.get("nll", np.nan)
        tw_nll = tw.get("nll", np.nan)
        delta_nll = tw_nll - scratch_nll if np.isfinite(scratch_nll) and np.isfinite(tw_nll) else np.nan
        if np.isfinite(s) and np.isfinite(delta_nll):
            s_list.append(s)
            delta_list.append(delta_nll)
        pairs_out.append({"target": tgt, "s_transfer": float(s), "delta_nll": float(delta_nll)})

    if len(s_list) < 2:
        return {
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "pairs": pairs_out,
            "n": len(s_list),
        }

    s_arr = np.array(s_list)
    d_arr = np.array(delta_list)
    pearson_r, pearson_p = stats.pearsonr(s_arr, d_arr)
    spearman_r, spearman_p = stats.spearmanr(s_arr, d_arr)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pairs": pairs_out,
        "n": len(s_list),
    }


def theorem_5_1_verification(
    stage5_report: dict,
    stage6_results: List[dict],
) -> dict:
    """
    Check Theorem 5.1 three conditions vs actual outcomes.
    Conditions (from paper_desc): (1) W_JD < W_crit, (2) rho_regime high, (3) spectral passed.
    Outcome: TransWeave success iff tw_nll <= scratch_nll (TW NLL more negative = better).

    Args:
        stage5_report: stage5_transfer_report.json
        stage6_results: stage6_experiment_results.json
    Returns:
        {"pairs": [{"target": str, "condition_1": bool, "condition_2": bool, "condition_3": bool,
                    "all_conditions": bool, "transweave_success": bool, "scratch_nll": float,
                    "transweave_nll": float}], "summary": {...}}
    """
    # Build stage5 lookup
    by_target = {}
    for p in stage5_report.get("pairs", []):
        tgt = p.get("target")
        if tgt:
            by_target[tgt] = {
                "w_jd_over_w_crit": p.get("w_jd_over_w_crit", 1.0),
                "rho": p.get("rho", 0),
                "spectral_passed": p.get("spectral", {}).get("passed", False),
            }

    pairs_out = []
    for row in stage6_results:
        tgt = row.get("target_asset")
        if not tgt:
            continue
        info = by_target.get(tgt, {})
        c1 = info.get("w_jd_over_w_crit", 1.0) < 1.0
        c2 = info.get("rho", 0) > 0.5
        c3 = info.get("spectral_passed", False)
        all_c = c1 and c2 and c3

        scratch_nll = row.get("scratch", {}).get("nll", np.nan)
        tw_nll = row.get("transweave", {}).get("nll", np.nan)
        # TransWeave success: TW NLL more negative (better) than Scratch. Lower NLL = better.
        success = tw_nll <= scratch_nll if (np.isfinite(tw_nll) and np.isfinite(scratch_nll)) else False

        pairs_out.append({
            "target": tgt,
            "condition_1_w_jd": c1,
            "condition_2_rho": c2,
            "condition_3_spectral": c3,
            "all_conditions": all_c,
            "transweave_success": success,
            "scratch_nll": float(scratch_nll),
            "transweave_nll": float(tw_nll),
        })

    n_success = sum(1 for p in pairs_out if p["transweave_success"])
    n_total = len(pairs_out)
    return {
        "pairs": pairs_out,
        "summary": {
            "n_pairs": n_total,
            "n_transweave_success": n_success,
            "success_rate": n_success / n_total if n_total > 0 else np.nan,
        },
    }
