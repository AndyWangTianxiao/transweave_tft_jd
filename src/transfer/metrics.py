"""
Transfer feasibility metrics: W_JD (formula 12), rho_regime (formula 13),
S_transfer (formula 58), spectral condition (formula 21), time change (formula 28-29).
Per doc/stage5_regime.md: paper formula (12) 3-term structure, W_crit = 2√(σ̄²+λ̄).
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


def _theta_to_mu_j(theta: dict) -> float:
    """Get mu_J from theta (handles mu_J / mu_j)."""
    return theta.get("mu_J", theta.get("mu_j", 0.0))


def _theta_to_sigma_j(theta: dict) -> float:
    """Get sigma_J from theta (handles sigma_J / sigma_j)."""
    return theta.get("sigma_J", theta.get("sigma_j", 0.0))


def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def _theta_to_lam(theta: dict) -> float:
    """Get lambda from theta (JSON uses 'lambda', code uses 'lam')."""
    return theta.get("lam", theta.get("lambda", 0.0))


def compute_v_j(theta: dict) -> float:
    """
    Jump variance contribution: v_J = λ · (μ_J² + σ_J²).
    Diagnostic only (v3: not used in decision). Per doc/stage5_regime.md.
    """
    lam = _theta_to_lam(theta)
    mu_j = _theta_to_mu_j(theta)
    sigma_j = _theta_to_sigma_j(theta)
    return lam * (mu_j ** 2 + sigma_j ** 2)


def compute_w2_jump(theta_a: dict, theta_b: dict) -> float:
    """
    W₂²(λ₁·P_φ₁, λ₂·P_φ₂) approximation. Per doc/stage5_regime.md Section 4.1.

    For λ₁ ≈ λ₂ (crypto same-domain): mass_term ≈ 0, W₂² ≈ λ̄·[(μ_J₁-μ_J₂)²+(σ_J₁-σ_J₂)²].
    """
    lam_a = _theta_to_lam(theta_a)
    lam_b = _theta_to_lam(theta_b)
    lam_bar = (lam_a + lam_b) / 2
    mass_term = (lam_a - lam_b) ** 2
    w2_shape = (_theta_to_mu_j(theta_a) - _theta_to_mu_j(theta_b)) ** 2 + (
        _theta_to_sigma_j(theta_a) - _theta_to_sigma_j(theta_b)
    ) ** 2
    return mass_term + lam_bar * w2_shape


def compute_w_jd(
    theta_a: dict,
    theta_b: dict,
    method: Literal["paper", "semantic"] = "paper",
) -> float:
    """
    Jump-Diffusion Wasserstein distance, formula (12).
    Primary (paper): 3-term structure per doc/stage5_regime.md.
    Diagnostic (semantic): legacy lambda-weighted version for notebook comparison.

    Args:
        theta_a, theta_b: Dicts with mu, sigma, lam/lambda, mu_J, sigma_J (annualized).
        method: "paper" (default) or "semantic"
    Returns:
        W_JD scalar.
    """
    if method == "paper":
        term_mu = (theta_a["mu"] - theta_b["mu"]) ** 2
        term_sigma = (theta_a["sigma"] - theta_b["sigma"]) ** 2
        term_jump = compute_w2_jump(theta_a, theta_b)
        return math.sqrt(term_mu + term_sigma + term_jump)

    # semantic: diagnostic only, doc/stage5_regime.md 3.1
    lam_a = _theta_to_lam(theta_a)
    lam_b = _theta_to_lam(theta_b)
    lam_bar = (lam_a + lam_b) / 2
    d_mu = (theta_a["mu"] - theta_b["mu"]) ** 2
    d_sigma = (theta_a["sigma"] - theta_b["sigma"]) ** 2
    d_lam = (lam_a - lam_b) ** 2
    d_jump = lam_bar * (
        (_theta_to_mu_j(theta_a) - _theta_to_mu_j(theta_b)) ** 2
        + (_theta_to_sigma_j(theta_a) - _theta_to_sigma_j(theta_b)) ** 2
    )
    return math.sqrt(d_mu + d_sigma + d_lam + d_jump)


def compute_w_crit(theta_seq_a: list, theta_seq_b: list) -> float:
    """
    Theorem 5.1 condition 1: W_crit = 2√(σ̄² + λ̄).
    Paper original formula, no substitution. Per doc/stage5_regime.md Section 4.2.
    """
    if not theta_seq_a or not theta_seq_b:
        return 1e-10
    n = min(len(theta_seq_a), len(theta_seq_b))
    sigma_a = np.mean([theta_seq_a[i]["sigma"] for i in range(n)])
    sigma_b = np.mean([theta_seq_b[i]["sigma"] for i in range(n)])
    lam_a = np.mean([_theta_to_lam(theta_seq_a[i]) for i in range(n)])
    lam_b = np.mean([_theta_to_lam(theta_seq_b[i]) for i in range(n)])
    sigma_bar = (sigma_a + sigma_b) / 2
    lam_bar = (lam_a + lam_b) / 2
    return max(2.0 * math.sqrt(sigma_bar ** 2 + lam_bar), 1e-10)


def compute_rho_regime(
    V_a: np.ndarray,
    V_b: np.ndarray,
    eigenvalues_a: Optional[np.ndarray] = None,
    eigenvalues_b: Optional[np.ndarray] = None,
    exclude_stationary: Optional[bool] = None,
    config_path: str = "config.yaml",
) -> Dict[str, float]:
    """
    Regime overlap coefficient, formula (13). v2: exclude stationary, no null-baseline.
    Returns rho (used directly for decision). Per doc/stage5_regime.md Section 4.3.
    """
    cfg = _load_config(config_path)
    if exclude_stationary is None:
        exclude_stationary = cfg.get("transfer", {}).get("rho_exclude_stationary", True)

    Va = np.real(np.real_if_close(np.asarray(V_a)))
    Vb = np.real(np.real_if_close(np.asarray(V_b)))
    n_a, n_b = Va.shape[1], Vb.shape[1]

    if exclude_stationary and eigenvalues_a is not None and eigenvalues_b is not None:
        ev_a = np.asarray(eigenvalues_a).ravel()
        ev_b = np.asarray(eigenvalues_b).ravel()
        idx_a = int(np.argmin(np.abs(ev_a - 1.0)))
        idx_b = int(np.argmin(np.abs(ev_b - 1.0)))
        keep_a = [i for i in range(n_a) if i != idx_a]
        keep_b = [i for i in range(n_b) if i != idx_b]
        Va = Va[:, keep_a]
        Vb = Vb[:, keep_b]
        n_a, n_b = Va.shape[1], Vb.shape[1]
        if n_a == 0 or n_b == 0:
            return {"rho": 0.0}

    n_min = min(n_a, n_b)

    # Pad rows (state dim) so Va and Vb have same n_rows for inner product
    n_rows_a, n_rows_b = Va.shape[0], Vb.shape[0]
    if n_rows_a < n_rows_b:
        Va = np.vstack([Va, np.zeros((n_rows_b - n_rows_a, Va.shape[1]))])
    elif n_rows_b < n_rows_a:
        Vb = np.vstack([Vb, np.zeros((n_rows_a - n_rows_b, Vb.shape[1]))])

    # Pad columns (vector count) if rectangular for Hungarian matching
    if n_a < n_b:
        Va = np.hstack([Va, np.zeros((Va.shape[0], n_b - n_a))])
    elif n_b < n_a:
        Vb = np.hstack([Vb, np.zeros((Vb.shape[0], n_a - n_b))])

    C = np.abs(np.conjugate(Va.T) @ Vb)
    C = np.real_if_close(C)
    row_ind, col_ind = linear_sum_assignment(1 - C)
    rho = float(C[row_ind, col_ind].sum() / n_min)
    rho = float(np.clip(rho, 0.0, 1.0))

    return {"rho": rho}


def compute_s_transfer(
    w_jd: float,
    rho_regime: float,
    delta_w_pt: float,
    w_crit: Optional[float] = None,
    theta_a: Optional[dict] = None,
    theta_b: Optional[dict] = None,
    sigma_bar: Optional[float] = None,
    lam_bar: Optional[float] = None,
) -> Dict[str, float]:
    """
    Transfer Readiness Score, formula (58).
    S = exp(-W_JD/W_crit) * rho_regime * exp(-|ΔW_PT|/0.2)
    When w_crit provided (M2 standardized), use directly. Else compute from sigma_bar/lam_bar.
    """
    if w_crit is None:
        if sigma_bar is None or lam_bar is None:
            if theta_a is not None and theta_b is not None:
                sigma_bar = (theta_a["sigma"] + theta_b["sigma"]) / 2
                lam_bar = (_theta_to_lam(theta_a) + _theta_to_lam(theta_b)) / 2
            else:
                sigma_bar = sigma_bar or 0.0
                lam_bar = lam_bar or 0.0
        w_crit = 2 * math.sqrt(sigma_bar ** 2 + lam_bar)
    w_crit = max(w_crit, 1e-10)
    s = math.exp(-w_jd / w_crit) * rho_regime * math.exp(-abs(delta_w_pt) / 0.2)
    return {"s_transfer": float(s), "w_crit": w_crit}


def spectral_transfer_condition(
    Lambda_a: np.ndarray,
    Lambda_b: np.ndarray,
    rho_regime: float,
    n: int,
    delta: float = 0.05,
) -> Dict[str, Any]:
    """
    Spectral transfer condition, formula (21). Diagnostic only.
    LHS = rho_regime * (1 - ||Λ_a - Λ_b||_F / (||Λ_a||_F + ||Λ_b||_F))
    For different n_states, compare first min(n_a, n_b) eigenvalues.
    """
    tau_crit = 0.5 + math.sqrt(math.log(1 / delta) / (2 * n))
    La = np.abs(np.asarray(Lambda_a).ravel())
    Lb = np.abs(np.asarray(Lambda_b).ravel())
    k = min(len(La), len(Lb))
    La_k, Lb_k = La[:k], Lb[:k]
    norm_diff = np.linalg.norm(La_k - Lb_k)
    norm_sum = np.linalg.norm(La_k) + np.linalg.norm(Lb_k) + 1e-10
    lhs = rho_regime * (1 - norm_diff / norm_sum)
    return {"lhs": lhs, "tau_crit": tau_crit, "passed": lhs > tau_crit}


def time_change_diagnostic(
    theta_a: dict,
    theta_b: dict,
    ratio_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Theorem 3.3, formulas (28-29). For static params, ΔΛ/T = |λ_a - λ_b|.
    Algorithm 1 Phase 3: ratio > 0.5 triggers time transform.
    """
    lam_a = _theta_to_lam(theta_a)
    lam_b = _theta_to_lam(theta_b)
    lam_min = min(lam_a, lam_b) + 1e-10
    delta_lam_ratio = abs(lam_a - lam_b) / lam_min
    needs_time_change = delta_lam_ratio > ratio_threshold
    condition_29 = abs(lam_a - lam_b) < min(lam_a, lam_b)
    return {
        "lam_a": lam_a,
        "lam_b": lam_b,
        "delta_lam_ratio": delta_lam_ratio,
        "needs_time_change": needs_time_change,
        "condition_29_satisfied": condition_29,
    }


def time_change_diagnostic_from_sequences(
    theta_seq_a: list,
    theta_seq_b: list,
    ratio_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    M6: Theorem 3.3 for rolling params. λ̄ = mean(λ_t) over windows.
    Diagnostic only, not used in Go/No-Go.
    """
    if not theta_seq_a or not theta_seq_b:
        return {"lam_a_mean": 0.0, "lam_b_mean": 0.0, "delta_lam_ratio": 0.0,
                "needs_time_change": False, "condition_29_satisfied": True}
    lam_a = np.mean([_theta_to_lam(t) for t in theta_seq_a])
    lam_b = np.mean([_theta_to_lam(t) for t in theta_seq_b])
    lam_min = min(lam_a, lam_b) + 1e-10
    delta_lam_ratio = abs(lam_a - lam_b) / lam_min
    return {
        "lam_a_mean": float(lam_a),
        "lam_b_mean": float(lam_b),
        "delta_lam_ratio": float(delta_lam_ratio),
        "needs_time_change": delta_lam_ratio > ratio_threshold,
        "condition_29_satisfied": delta_lam_ratio < 1.0,
    }


def regime_diagnostics(
    P_a: np.ndarray,
    P_b: np.ndarray,
    means_a: np.ndarray,
    means_b: np.ndarray,
) -> Dict[str, Any]:
    """
    Regime diagnostics per doc/stage5_regime.md Section 10 Priority 5.
    - Steady-state distribution: π = πP, compare JS(π_a, π_b)
    - Aligned transition matrix diff: d_P = ||P_a - Π^T P_b Π||_F after state alignment
    """
    # Steady-state: left eigenvector of P with eigenvalue 1
    ev_a, evl_a = np.linalg.eig(P_a.T)
    ev_a = np.real_if_close(ev_a)
    evl_a = np.real_if_close(evl_a)
    idx_a = np.argmin(np.abs(ev_a - 1.0))
    pi_a = np.real(evl_a[:, idx_a])
    pi_a = np.maximum(pi_a, 0)
    sa = pi_a.sum()
    pi_a = pi_a / sa if sa > 1e-10 else np.ones_like(pi_a) / len(pi_a)

    ev_b, evl_b = np.linalg.eig(P_b.T)
    ev_b = np.real_if_close(ev_b)
    evl_b = np.real_if_close(evl_b)
    idx_b = np.argmin(np.abs(ev_b - 1.0))
    pi_b = np.real(evl_b[:, idx_b])
    pi_b = np.maximum(pi_b, 0)
    sb = pi_b.sum()
    pi_b = pi_b / sb if sb > 1e-10 else np.ones_like(pi_b) / len(pi_b)

    # L1 distance
    n_a, n_b = len(pi_a), len(pi_b)
    if n_a != n_b:
        # Pad to max
        pi_a_pad = np.zeros(max(n_a, n_b))
        pi_b_pad = np.zeros(max(n_a, n_b))
        pi_a_pad[:n_a] = pi_a
        pi_b_pad[:n_b] = pi_b
        pi_a, pi_b = pi_a_pad, pi_b_pad
    pi_diff_l1 = float(np.sum(np.abs(pi_a - pi_b)))

    # JS divergence (discrete)
    eps = 1e-10
    m = (pi_a + pi_b) / 2
    m = m + eps
    pi_a = pi_a + eps
    pi_b = pi_b + eps
    js = 0.5 * (np.sum(pi_a * np.log(pi_a / m)) + np.sum(pi_b * np.log(pi_b / m)))
    js_pi = float(js)

    # Aligned d_P: match states by emission means (2D: return, rv)
    # Simple: match by mean[0] (return) ordering
    if means_a.shape[0] >= 2 and means_b.shape[0] >= 2:
        order_a = np.argsort(means_a[:, 0])
        order_b = np.argsort(means_b[:, 0])
        P_a_ord = P_a[order_a][:, order_a]
        P_b_ord = P_b[order_b][:, order_b]
        k = min(P_a_ord.shape[0], P_b_ord.shape[0])
        d_P = np.linalg.norm(P_a_ord[:k, :k] - P_b_ord[:k, :k], "fro")
    else:
        d_P = float("nan")

    return {
        "pi_a": pi_a,
        "pi_b": pi_b,
        "pi_diff_l1": pi_diff_l1,
        "js_pi": js_pi,
        "d_P": d_P,
    }


def _rolling_sigma_lam_bar(theta_seq_a: list, theta_seq_b: list) -> tuple:
    """Mean sigma and lam over aligned rolling theta sequences for w_crit."""
    if not theta_seq_a or not theta_seq_b:
        return 0.0, 0.0
    n = min(len(theta_seq_a), len(theta_seq_b))
    sigmas = [(theta_seq_a[i]["sigma"] + theta_seq_b[i]["sigma"]) / 2 for i in range(n)]
    lams = [(_theta_to_lam(theta_seq_a[i]) + _theta_to_lam(theta_seq_b[i])) / 2 for i in range(n)]
    return float(np.mean(sigmas)), float(np.mean(lams))


def compute_rolling_w_jd(theta_seq_a: list, theta_seq_b: list) -> dict:
    """
    Rolling W_JD per paper formula (12). p90 for decision. No burstiness (v3).
    Per doc/stage5_regime.md Section 4.4.
    """
    if not theta_seq_a or not theta_seq_b:
        return {
            "w_jd_effective": 0.0,
            "w_jd_p90": 0.0,
            "w_jd_mean": 0.0,
            "n_windows": 0,
            "w_jd_per_window": [],
        }

    n = min(len(theta_seq_a), len(theta_seq_b))
    w_list = [compute_w_jd(theta_seq_a[i], theta_seq_b[i], method="paper") for i in range(n)]
    w_p90 = float(np.quantile(w_list, 0.9))
    w_mean = float(np.mean(w_list))

    return {
        "w_jd_effective": w_p90,
        "w_jd_p90": w_p90,
        "w_jd_mean": w_mean,
        "n_windows": n,
        "w_jd_per_window": w_list,
    }


def make_transfer_decision(
    w_jd: float,
    w_crit: float,
    rho: float,
    delta_wpt: float,
    s_transfer: Optional[float] = None,
    s_transfer_adjusted: Optional[float] = None,
    w_jd_effective: Optional[float] = None,
    config_path: str = "config.yaml",
    return_dict: bool = True,
):
    """
    Algorithm 1 Phase 1-2 decision logic. v2: use rho (exclude stationary, no null).
    Returns dict with decision, reason when return_dict=True; else legacy string.
    """
    config = _load_config(config_path)
    xfer = config.get("transfer", {})
    rho_reject = xfer.get("rho_reject", 0.3)
    rho_full = xfer.get("rho_full", 0.7)
    rho_partial = xfer.get("rho_partial", 0.5)
    delta_wpt_full = xfer.get("delta_wpt_full", 0.1)
    s_threshold = xfer.get("s_tiebreaker_threshold", 0.5)
    use_s_adj = xfer.get("use_s_adjusted_in_decision", False)

    w_jd_check = w_jd_effective if w_jd_effective is not None else w_jd
    s_eff = s_transfer
    if use_s_adj and s_transfer_adjusted is not None:
        s_eff = s_transfer_adjusted

    if w_jd_check > w_crit:
        out = {"decision": "reject", "reason": f"W_JD({w_jd_check:.3f}) > W_crit({w_crit:.3f})"}
        return out if return_dict else out["decision"]
    if rho < rho_reject:
        out = {"decision": "reject", "reason": f"rho({rho:.3f}) < rho_reject"}
        return out if return_dict else out["decision"]
    if rho > rho_full and delta_wpt < delta_wpt_full:
        decision = "full"
        if s_eff is not None and s_eff < s_threshold:
            decision = "partial"
        out = {"decision": decision, "reason": f"rho={rho:.3f}, dwpt={delta_wpt:.4f}, S={s_eff:.3f}"}
        return out if return_dict else out["decision"]
    if rho > rho_partial:
        out = {"decision": "partial", "reason": f"rho={rho:.3f} > rho_partial"}
        return out if return_dict else out["decision"]
    out = {"decision": "weak", "reason": f"rho={rho:.3f} < rho_partial"}
    return out if return_dict else out["decision"]
