"""
GBM (Geometric Brownian Motion) baseline: pure Gaussian, constant (mu, sigma, lambda=0).
Per doc/stage8a_audit_present.md Task 2.4.
"""

from typing import Dict

import numpy as np
import torch

from ..models import jump_diffusion as jd
from ..models import losses


def evaluate_gbm_baseline(
    r_train: np.ndarray,
    r_test: np.ndarray,
    dt: float,
    m_crps: int = 500,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Pure Gaussian: constant (mu, sigma, lambda=0). Returns nll, crps.

    Args:
        r_train: (T,) train returns for MLE.
        r_test: (T,) test returns for evaluation.
        dt: 1 / bars_per_year.
        m_crps: Monte Carlo samples for CRPS.
        seed: random seed.
    Returns:
        {"nll": float, "crps": float, "mu": float, "sigma": float}
    """
    r_train = np.asarray(r_train, dtype=np.float64).ravel()
    r_test = np.asarray(r_test, dtype=np.float64).ravel()
    # Drop NaN/Inf for robust MLE
    r_train = r_train[np.isfinite(r_train)]
    r_test = r_test[np.isfinite(r_test)]
    if len(r_train) < 10 or len(r_test) < 10:
        return {"nll": np.nan, "crps": np.nan, "mu": np.nan, "sigma": np.nan}
    # GBM: r ~ N(mu*dt, sigma^2*dt). MLE: mu_hat = mean(r)/dt, sigma_hat = std(r)/sqrt(dt)
    mu_hat = float(np.mean(r_train)) / dt
    sigma_hat = float(np.std(r_train) + 1e-10) / np.sqrt(dt)
    lam = 0.0
    mu_J = 0.0
    sigma_J = 1e-6  # negligible

    r_t = torch.tensor(r_test, dtype=torch.float64)
    n_max = 10
    nll_val = -jd.log_density(r_t, mu_hat, sigma_hat, lam, mu_J, sigma_J, dt, n_max).mean().item()
    crps_val = losses.crps_mc(
        r_t, mu_hat, sigma_hat, lam, mu_J, sigma_J, dt, m_crps, seed,
    ).item()
    return {
        "nll": float(nll_val),
        "crps": float(crps_val),
        "mu": float(mu_hat),
        "sigma": float(sigma_hat),
    }
