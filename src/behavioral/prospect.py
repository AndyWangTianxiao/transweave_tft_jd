"""
Prospect theory functions per formula (35).
π(p): Prelec probability weighting; v(m): value function (gain/loss).
Input p must be probability in (0, 1], NOT λΔt.
"""

from pathlib import Path
from typing import Union

import torch
import yaml


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load config from project root."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def prelec_weight(p: torch.Tensor, delta: Union[float, None] = None) -> torch.Tensor:
    """
    Prelec probability weighting: π(p) = exp(-(-ln p)^δ). Formula (35).

    Args:
        p: Probability in (0, 1]. Caller must pass probability (e.g. 1-exp(-λΔt)), NOT λΔt.
        delta: Curvature parameter. If None, read from config.prospect_theory.delta.
    Returns:
        π(p), same shape as p.
    """
    if delta is None:
        cfg = _load_config()
        delta = cfg["prospect_theory"]["delta"]
    p = p.clamp(min=1e-10, max=1.0)
    log_p = torch.log(p)
    # (-ln p)^δ; -ln p >= 0 when p <= 1
    inner = torch.pow(-log_p, delta)
    return torch.exp(-inner)


def value_function(
    m: torch.Tensor,
    alpha: Union[float, None] = None,
    beta: Union[float, None] = None,
    lambda_loss: Union[float, None] = None,
) -> torch.Tensor:
    """
    Prospect theory value function. Formula (35).
    v(m) = m^α for m >= 0 (gain), v(m) = -λ·(-m)^β for m < 0 (loss).

    Args:
        m: Outcome (e.g. μ_t - λ_t·E[Y]). Can be positive or negative.
        alpha, beta, lambda_loss: If None, read from config.prospect_theory.
    Returns:
        v(m), same shape as m. Can be positive or negative.
    """
    if alpha is None or beta is None or lambda_loss is None:
        cfg = _load_config()
        pt = cfg["prospect_theory"]
        alpha = alpha if alpha is not None else pt["alpha"]
        beta = beta if beta is not None else pt["beta"]
        lambda_loss = lambda_loss if lambda_loss is not None else pt["lambda_loss"]

    gain = m >= 0
    loss = ~gain
    out = torch.zeros_like(m)
    out[gain] = torch.pow(m[gain].clamp(min=1e-10), alpha)
    out[loss] = -lambda_loss * torch.pow((-m[loss]).clamp(min=1e-10), beta)
    return out
