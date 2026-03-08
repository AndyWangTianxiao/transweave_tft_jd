"""
Weakness regularizer per formulas (43), (44).
Stage 4 implements only the first term of formula (50): -log W_PT(θ).
The second term (migration weakness consistency) is added in Stage 6/7.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import yaml

from .weakness import compute_w_pt


def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def _dim_eff_simple(lam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Simplified dim_eff per formula (44): 1 + Var(λ_t) / E[λ_t].
    Full formula has tr(I(θ)^{1/2}) factor; we omit for TFT-JD high-dim.
    """
    mean_lam = lam.mean()
    var_lam = lam.var()
    return 1.0 + var_lam / (mean_lam + eps)


def weakness_regularizer(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    lam: torch.Tensor,
    mu_J: torch.Tensor,
    sigma_J: torch.Tensor,
    dt: float,
    lambda_weak: Optional[float] = None,
    alpha_dim: Optional[float] = None,
    cvar_method: str = "analytic",
    config_path: str = "config.yaml",
) -> torch.Tensor:
    """
    L_weakness = lambda_weak · (-log W_PT) + alpha_dim · dim_eff.
    Formula (43) with β merged into lambda_weak.

    Stage 4: only -log W_PT term. Formula (50) second term (migration consistency) in Stage 6/7.

    Returns:
        Scalar loss tensor.
    """
    cfg = _load_config(config_path)
    w_cfg = cfg.get("weakness", {})
    lambda_weak = lambda_weak if lambda_weak is not None else w_cfg.get("lambda_weak", 0.01)
    alpha_dim = alpha_dim if alpha_dim is not None else w_cfg.get("alpha_dim", 0.001)

    w_pt = compute_w_pt(
        mu, sigma, lam, mu_J, sigma_J, dt,
        cvar_method=cvar_method,
        config_path=config_path,
    )
    neg_log_w = -torch.log(w_pt)
    term1 = lambda_weak * neg_log_w.mean()
    dim_eff = _dim_eff_simple(lam)
    term2 = alpha_dim * dim_eff
    return term1 + term2
