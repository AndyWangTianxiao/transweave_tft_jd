"""
Loss functions for jump-diffusion: NLL (formula 8), CRPS via Monte Carlo (no closed form), combined L (formula 7).
CRPS for JD mixture has no analytic solution; we sample M paths from p(r|θ) and use empirical CDF.
"""

from pathlib import Path
from typing import Union

import torch
import yaml

from . import jump_diffusion as jd


def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def nll(
    r: torch.Tensor,
    mu: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
    lam: Union[float, torch.Tensor],
    mu_J: Union[float, torch.Tensor],
    sigma_J: Union[float, torch.Tensor],
    dt: float,
    n_max: int,
) -> torch.Tensor:
    """
    Negative log-likelihood: -Σ_t log p(r_t|θ_t). Formula (7) first term.
    Supports static (scalar params) or per-step (params with same leading dim as r).
    Returns scalar (sum over r).
    """
    log_p = jd.log_density(r, mu, sigma, lam, mu_J, sigma_J, dt, n_max)
    return -log_p.sum()


def nll_clamped(
    r: torch.Tensor,
    mu: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
    lam: Union[float, torch.Tensor],
    mu_J: Union[float, torch.Tensor],
    sigma_J: Union[float, torch.Tensor],
    dt: float,
    n_max: int,
    clamp_sigma: float = 5.0,
) -> torch.Tensor:
    """
    Per-sample NLL with clamp at mean + clamp_sigma * std to prevent extreme bars from
    dominating gradients. Robust to inf/nan in nll_per (TASK_FIX_PHASE4 Section 3.2).
    Returns scalar (mean of clamped per-sample NLL). Per doc/stage6_transfer.md Section 14.4:
    sum caused L_JD ~±8000 at batch=2048, drowning L_TransWeave; mean yields ~±4.
    """
    log_p = jd.log_density(r, mu, sigma, lam, mu_J, sigma_J, dt, n_max)
    nll_per = -log_p
    finite_mask = torch.isfinite(nll_per)
    n_finite = finite_mask.sum()

    if n_finite == 0:
        return torch.tensor(50.0, device=r.device, dtype=nll_per.dtype)

    nll_finite = nll_per[finite_mask]
    mu_val = nll_finite.mean().detach()
    sigma_val = nll_finite.std().detach().clamp(min=1e-6)
    thresh = mu_val + clamp_sigma * sigma_val
    nll_clipped = torch.where(finite_mask, torch.clamp(nll_per, max=thresh), thresh)
    return nll_clipped.mean()


def crps_mc(
    r_obs: torch.Tensor,
    mu: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
    lam: Union[float, torch.Tensor],
    mu_J: Union[float, torch.Tensor],
    sigma_J: Union[float, torch.Tensor],
    dt: float,
    m_samples: int,
    seed: int,
    device: Union[str, torch.device] = None,
) -> torch.Tensor:
    """
    CRPS via Monte Carlo: JD mixture has no closed-form CRPS. Sample M from p(r|θ),
    then CRPS(F̂, y) = E|X - y| - 0.5 E|X - X'| ≈ (1/M)Σ|r_m - y| - (1/(2M²))Σ_m Σ_m' |r_m - r_m'|.
    r_obs: (batch,) observed returns.
    Returns scalar (mean CRPS over batch).
    """
    if device is None:
        device = r_obs.device
    r_obs = r_obs.to(device).double()
    batch = r_obs.numel()
    r_flat = r_obs.reshape(-1)

    # Static params: one set of M*batch samples, reshape to (batch, M)
    if isinstance(mu, (int, float)) and isinstance(sigma, (int, float)):
        samples = jd.sample_one_step(
            float(mu), float(sigma), float(lam), float(mu_J), float(sigma_J),
            dt, batch * m_samples, seed, device,
        )
        samples = samples.reshape(batch, m_samples).double()
        term1 = (samples - r_flat.unsqueeze(1)).abs().mean(dim=1)
        term2 = (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean(dim=(1, 2))
        crps_per = term1 - 0.5 * term2
        return crps_per.mean().float()
    # Per-step params: loop (or batch later for Stage 3)
    crps_list = []
    for i in range(batch):
        y = r_flat[i].item()
        mu_i = mu if isinstance(mu, (int, float)) else mu.reshape(-1)[min(i, mu.numel() - 1)].item()
        sigma_i = sigma if isinstance(sigma, (int, float)) else sigma.reshape(-1)[min(i, sigma.numel() - 1)].item()
        lam_i = lam if isinstance(lam, (int, float)) else lam.reshape(-1)[min(i, lam.numel() - 1)].item()
        mu_J_i = mu_J if isinstance(mu_J, (int, float)) else mu_J.reshape(-1)[min(i, mu_J.numel() - 1)].item()
        sigma_J_i = sigma_J if isinstance(sigma_J, (int, float)) else sigma_J.reshape(-1)[min(i, sigma_J.numel() - 1)].item()
        s = jd.sample_one_step(mu_i, sigma_i, lam_i, mu_J_i, sigma_J_i, dt, m_samples, seed + i, device).double()
        term1 = (s - y).abs().mean()
        idx = torch.randperm(m_samples, device=device)
        term2 = (s - s[idx]).abs().mean()
        crps_list.append((term1 - 0.5 * term2).item())
    return torch.tensor(sum(crps_list) / len(crps_list), device=device, dtype=torch.float32)


def combined_loss(
    r: torch.Tensor,
    mu: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
    lam: Union[float, torch.Tensor],
    mu_J: Union[float, torch.Tensor],
    sigma_J: Union[float, torch.Tensor],
    dt: float,
    n_max: int,
    alpha: float,
    beta: float,
    m_crps: int = 200,
    seed: int = 42,
    r_reg: float = 0.0,
) -> torch.Tensor:
    """
    Combined loss formula (7): L = NLL + α·CRPS + β·R.
    R is regularization (Stage 3); Stage 2 use R=0.
    """
    nll_val = nll(r, mu, sigma, lam, mu_J, sigma_J, dt, n_max)
    crps_val = crps_mc(r, mu, sigma, lam, mu_J, sigma_J, dt, m_crps, seed)
    return nll_val + alpha * crps_val + beta * r_reg
