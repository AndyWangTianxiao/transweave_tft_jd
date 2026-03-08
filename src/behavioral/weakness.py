"""
Prospect-theoretic weakness W_PT(θ_t) per formulas (36), (37).
8-step flow with P0 fixes: p_jump=1-exp(-λΔt), softplus+batch-max normalization.
CVaR: analytic heuristic (training) and MC (validation).
"""

import math
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .prospect import prelec_weight, value_function


def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def cvar_analytic_heuristic(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lam: torch.Tensor,
    mu_J: torch.Tensor,
    sigma_J: torch.Tensor,
    dt: float,
    alpha: float = 0.05,
    n_max: int = 10,
) -> torch.Tensor:
    """
    Mixture-CVaR heuristic (differentiable, fast). For training.
    Returns return convention: negative = big loss. Caller converts to loss: cvar_loss = -cvar_return.

    Per-component: CVaR_α(N(μ,σ²)) = μ - σ · φ(Φ⁻¹(α)) / α.
    Weights: truncated Poisson, softmax normalized.
    """
    device = mu.device
    dtype = mu.dtype
    lam_dt = (lam * dt).clamp(min=1e-10)
    # log_weights: n*log(λΔt) - λΔt - log(n!)
    ns = torch.arange(n_max + 1, dtype=dtype, device=device)
    log_n_factorial = torch.lgamma(ns + 1)
    # (batch,) -> (batch, n_max+1)
    log_weights = ns.unsqueeze(0) * torch.log(lam_dt).unsqueeze(1) - lam_dt.unsqueeze(1) - log_n_factorial.unsqueeze(0)
    weights = F.softmax(log_weights, dim=-1)

    # comp_mu = μ·Δt + n·μ_J, comp_var = σ²·Δt + n·σ_J²
    comp_mu = mu.unsqueeze(1) * dt + ns.unsqueeze(0) * mu_J.unsqueeze(1)
    comp_var = (sigma ** 2).unsqueeze(1) * dt + ns.unsqueeze(0) * (sigma_J ** 2).unsqueeze(1)
    comp_sigma = torch.sqrt(comp_var.clamp(min=1e-10))

    # Φ⁻¹(α): inverse CDF of standard normal. Φ⁻¹(α) = √2 · erfinv(2α - 1)
    x = max(-0.9999, min(0.9999, 2 * alpha - 1))
    z_alpha = math.sqrt(2) * torch.erfinv(torch.tensor(x, dtype=dtype, device=device)).item()
    phi_z = math.exp(-0.5 * z_alpha ** 2) / math.sqrt(2 * math.pi)
    comp_cvar = comp_mu - comp_sigma * (phi_z / alpha)
    cvar_return = (weights * comp_cvar).sum(dim=-1)
    return cvar_return


def cvar_mc(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lam: torch.Tensor,
    mu_J: torch.Tensor,
    sigma_J: torch.Tensor,
    dt: float,
    alpha: float = 0.05,
    n_samples: int = 5000,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    MC CVaR (validation only, not in training loop).
    Returns return convention: negative = big loss. Caller converts: cvar_loss = -cvar_return.
    """
    B = mu.shape[0]
    device = mu.device
    dtype = mu.dtype
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    lam_dt = (lam * dt).unsqueeze(1).expand(B, n_samples)
    eps = torch.randn(B, n_samples, device=device, dtype=dtype, generator=gen)
    n_jumps = torch.poisson(lam_dt.clamp(min=1e-10), generator=gen)
    jump_mean = n_jumps * mu_J.unsqueeze(1)
    jump_std = torch.sqrt((n_jumps * (sigma_J.unsqueeze(1) ** 2)).clamp(min=1e-10))
    jump = jump_mean + jump_std * torch.randn(B, n_samples, device=device, dtype=dtype, generator=gen)
    x = mu.unsqueeze(1) * dt + sigma.unsqueeze(1) * math.sqrt(dt) * eps + jump

    k = max(1, int(math.ceil(alpha * n_samples)))
    tail, _ = torch.topk(x, k, dim=1, largest=False)
    cvar_return = tail.mean(dim=1)
    return cvar_return


def compute_w_pt(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lam: torch.Tensor,
    mu_J: torch.Tensor,
    sigma_J: torch.Tensor,
    dt: float,
    gamma: Optional[float] = None,
    alpha: Optional[float] = None,
    cvar_method: Optional[Literal["analytic", "mc"]] = None,
    n_max: int = 10,
    cvar_mc_samples: Optional[int] = None,
    w_pt_eps: Optional[float] = None,
    config_path: str = "config.yaml",
    return_raw: bool = False,
) -> torch.Tensor:
    """
    W_PT(θ_t) per formula (37), 8-step flow with P0 fixes.

    Step 1: p_jump = 1 - exp(-λΔt)
    Step 2: pi_val = π(p_jump)
    Step 3: m = μ_t - λ_t·μ_J_t, v_val = v(m)
    Step 4-5: cvar_loss (loss convention), risk_factor = exp(-γ·cvar_loss)
    Step 6: raw_score = pi_val * v_val * risk_factor
    Step 7: w_pt_raw = softplus(raw_score), w_pt = w_pt_raw / (max.detach() + eps), clamp [1e-8, 1] (P0.2)

    Args:
        return_raw: If True, return w_pt_raw (softplus(raw_score)) without batch-max normalization.
            Use for cross-model comparison with global normalization (Option C).

    Returns:
        W_PT, shape (batch,) or (T,). If return_raw=False: values in (0, 1]. If return_raw=True: raw scores.
    """
    cfg = _load_config(config_path)
    risk_cfg = cfg.get("risk", {})
    weakness_cfg = cfg.get("weakness", {})
    gamma = gamma if gamma is not None else risk_cfg.get("gamma", 7.0)
    alpha = alpha if alpha is not None else weakness_cfg.get("cvar_alpha", risk_cfg.get("cvar_alpha", 0.05))
    cvar_method = cvar_method if cvar_method is not None else weakness_cfg.get("cvar_method", "analytic")
    cvar_mc_samples = cvar_mc_samples if cvar_mc_samples is not None else weakness_cfg.get("cvar_mc_samples", 5000)
    w_pt_eps = w_pt_eps if w_pt_eps is not None else weakness_cfg.get("w_pt_eps", 1e-8)

    # Step 1: p_jump = 1 - exp(-λΔt) (P0.1)
    p_jump = 1.0 - torch.exp(-lam * dt)
    p_jump = p_jump.clamp(min=1e-10, max=1.0)

    # Step 2: Prelec
    pi_val = prelec_weight(p_jump)

    # Step 3: m = μ_t - λ_t·μ_J_t, v(m)
    m = mu - lam * mu_J
    v_val = value_function(m)

    # Step 4-5: CVaR in loss convention
    if cvar_method == "analytic":
        cvar_return = cvar_analytic_heuristic(mu, sigma, lam, mu_J, sigma_J, dt, alpha, n_max)
    else:
        cvar_return = cvar_mc(mu, sigma, lam, mu_J, sigma_J, dt, alpha, cvar_mc_samples)
    cvar_loss = -cvar_return  # convert to loss convention (positive = high risk)
    risk_factor = torch.exp(-gamma * cvar_loss)

    # Step 6: raw_score
    raw_score = pi_val * v_val * risk_factor

    # Step 7: softplus + batch-max normalization (P0.2)
    w_pt_raw = F.softplus(raw_score)
    if return_raw:
        return w_pt_raw
    w_max = w_pt_raw.max().detach() + w_pt_eps
    w_pt = w_pt_raw / w_max
    w_pt = w_pt.clamp(min=1e-8, max=1.0)
    return w_pt


def compute_w_pt_static(
    theta: dict,
    dt: float,
    config_path: str = "config.yaml",
) -> float:
    """
    W_PT for static JD parameters (Stage 5). Uses return_raw=True to avoid batch-max
    normalization, since we compare single-parameter W_PT across assets.

    Args:
        theta: Dict with keys mu, sigma, lam (or lambda), mu_J, sigma_J (annualized).
        dt: 1 / bars_per_year.
    Returns:
        Scalar W_PT value (raw score, not normalized).
    """
    lam = theta.get("lam", theta.get("lambda", 0.0))
    mu_t = torch.tensor([[theta["mu"]]], dtype=torch.float64)
    sigma_t = torch.tensor([[theta["sigma"]]], dtype=torch.float64)
    lam_t = torch.tensor([[lam]], dtype=torch.float64)
    mu_J_t = torch.tensor([[theta["mu_J"]]], dtype=torch.float64)
    sigma_J_t = torch.tensor([[theta["sigma_J"]]], dtype=torch.float64)
    w_raw = compute_w_pt(
        mu_t, sigma_t, lam_t, mu_J_t, sigma_J_t, dt,
        config_path=config_path, return_raw=True,
    )
    return float(w_raw.squeeze().item())


def compute_rolling_w_pt(
    theta_seq_a: list,
    theta_seq_b: list,
    dt: float,
    config_path: str = "config.yaml",
) -> dict:
    """
    Rolling W_PT per doc/stage5_regime.md 10.4 (legacy mean-based).
    For each theta in theta_seq, compute W_PT(θ) via compute_w_pt_static.
    W_PT_a = mean(W_PT over theta_seq_a), W_PT_b = mean(W_PT over theta_seq_b).
    delta_w_pt = |W_PT_a - W_PT_b| / (W_PT_a + W_PT_b + eps).

    Args:
        theta_seq_a, theta_seq_b: Lists of theta dicts from rolling MLE.
        dt: 1 / bars_per_year.
    Returns:
        Dict with w_pt_a, w_pt_b, delta_w_pt, n_windows.
    """
    if not theta_seq_a or not theta_seq_b:
        return {"w_pt_a": 0.0, "w_pt_b": 0.0, "delta_w_pt": 0.0, "n_windows": 0}
    n = min(len(theta_seq_a), len(theta_seq_b))
    w_pt_a_list = [compute_w_pt_static(theta_seq_a[i], dt, config_path) for i in range(n)]
    w_pt_b_list = [compute_w_pt_static(theta_seq_b[i], dt, config_path) for i in range(n)]
    w_pt_a = sum(w_pt_a_list) / n
    w_pt_b = sum(w_pt_b_list) / n
    eps = 1e-10
    delta_w_pt = abs(w_pt_a - w_pt_b) / (w_pt_a + w_pt_b + eps)
    return {
        "w_pt_a": w_pt_a,
        "w_pt_b": w_pt_b,
        "delta_w_pt": delta_w_pt,
        "n_windows": n,
    }


def compute_rolling_delta_wpt(
    theta_seq_a: list,
    theta_seq_b: list,
    dt: float,
    config_path: str = "config.yaml",
) -> dict:
    """
    M3: ΔW_PT with p90 aggregation. Per-window delta, then p90 as main metric.
    Returns delta_wpt (p90, for decision), delta_wpt_mean (diagnostic),
    wpt_a_list, wpt_b_list, delta_per_window.
    """
    if not theta_seq_a or not theta_seq_b:
        return {"delta_wpt": 0.0, "delta_wpt_mean": 0.0, "wpt_a_list": [], "wpt_b_list": [],
                "delta_per_window": [], "n_windows": 0}
    n = min(len(theta_seq_a), len(theta_seq_b))
    wpt_a_list = []
    wpt_b_list = []
    delta_list = []
    for i in range(n):
        wpt_a = compute_w_pt_static(theta_seq_a[i], dt, config_path)
        wpt_b = compute_w_pt_static(theta_seq_b[i], dt, config_path)
        wpt_a_list.append(wpt_a)
        wpt_b_list.append(wpt_b)
        denom = abs(wpt_a) + abs(wpt_b) + 1e-10
        delta_list.append(abs(wpt_a - wpt_b) / denom)
    delta_wpt_p90 = float(np.quantile(delta_list, 0.9))
    delta_wpt_mean = float(np.mean(delta_list))
    return {
        "delta_wpt": delta_wpt_p90,
        "delta_wpt_mean": delta_wpt_mean,
        "wpt_a_list": wpt_a_list,
        "wpt_b_list": wpt_b_list,
        "delta_per_window": delta_list,
        "n_windows": n,
    }


def compute_delta_wpt_ks(wpt_a_list: list, wpt_b_list: list) -> dict:
    """
    M3: KS statistic for W_PT distribution difference. Diagnostic only.
    """
    from scipy.stats import ks_2samp
    if not wpt_a_list or not wpt_b_list:
        return {"ks_stat": 0.0, "ks_pvalue": 1.0}
    stat, pvalue = ks_2samp(wpt_a_list, wpt_b_list)
    return {"ks_stat": float(stat), "ks_pvalue": float(pvalue)}
