"""
Calibration evaluation: PIT (Probability Integral Transform), NLL per sample.
Per doc/stage8a_audit_present.md Task 2.1. Uses JD CDF (formula 8).
"""

from typing import Union

import numpy as np
from scipy import stats


def _jd_cdf(
    r: np.ndarray,
    mu: Union[float, np.ndarray],
    sigma: Union[float, np.ndarray],
    lam: Union[float, np.ndarray],
    mu_J: Union[float, np.ndarray],
    sigma_J: Union[float, np.ndarray],
    dt: float,
    n_max: int,
) -> np.ndarray:
    """
    Jump-diffusion CDF via mixture of normals. Formula (8).
    F(r) = sum_n w_n * Phi((r - mean_n) / sqrt(var_n)).
    """
    r = np.asarray(r, dtype=np.float64).ravel()
    n = len(r)
    # Broadcast scalars to arrays
    mu = np.broadcast_to(mu, n) if np.isscalar(mu) or np.asarray(mu).size == 1 else np.asarray(mu).ravel()
    sigma = np.broadcast_to(sigma, n) if np.isscalar(sigma) or np.asarray(sigma).size == 1 else np.asarray(sigma).ravel()
    lam = np.broadcast_to(lam, n) if np.isscalar(lam) or np.asarray(lam).size == 1 else np.asarray(lam).ravel()
    mu_J = np.broadcast_to(mu_J, n) if np.isscalar(mu_J) or np.asarray(mu_J).size == 1 else np.asarray(mu_J).ravel()
    sigma_J = np.broadcast_to(sigma_J, n) if np.isscalar(sigma_J) or np.asarray(sigma_J).size == 1 else np.asarray(sigma_J).ravel()

    lam_dt = lam * dt
    drift_dt = mu * dt
    var_dt = (sigma ** 2) * dt

    # n=0: weight = exp(-lam_dt), mean = drift_dt, var = var_dt
    w0 = np.exp(-lam_dt)
    mean0 = drift_dt
    std0 = np.sqrt(np.maximum(var_dt, 1e-12))
    cdf = w0 * stats.norm.cdf((r - mean0) / std0)

    for k in range(1, n_max + 1):
        log_w = k * np.log(np.maximum(lam_dt, 1e-12)) - lam_dt - np.sum(np.log(np.arange(1, k + 1)))
        w_k = np.exp(log_w)
        mean_k = drift_dt + k * mu_J
        var_k = var_dt + k * (sigma_J ** 2)
        std_k = np.sqrt(np.maximum(var_k, 1e-12))
        cdf += w_k * stats.norm.cdf((r - mean_k) / std_k)

    return cdf


def compute_pit(
    r_actual: np.ndarray,
    theta_predicted: np.ndarray,
    dt: float,
    n_max: int = 10,
) -> np.ndarray:
    """
    Probability Integral Transform via JD CDF (Eq 8).
    PIT(r) = F_theta(r) where F is the JD CDF.

    Args:
        r_actual: (T,) observed returns.
        theta_predicted: (T, 5) or (5,) JD params (mu, sigma, lam, mu_J, sigma_J) per step.
        dt: 1 / bars_per_year.
        n_max: truncation for JD mixture.
    Returns:
        (T,) PIT values in [0, 1].
    """
    r_actual = np.asarray(r_actual, dtype=np.float64).ravel()
    theta = np.asarray(theta_predicted, dtype=np.float64)
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, (len(r_actual), 5))
    mu, sigma, lam, mu_J, sigma_J = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4]
    pit = _jd_cdf(r_actual, mu, sigma, lam, mu_J, sigma_J, dt, n_max)
    return np.clip(pit, 1e-8, 1 - 1e-8)


def pit_uniformity_test(pit_values: np.ndarray) -> dict:
    """
    KS test against Uniform(0, 1). Returns ks_stat, p_value.

    Args:
        pit_values: (T,) PIT values.
    Returns:
        {"ks_stat": float, "p_value": float, "n": int}
    """
    pit = np.asarray(pit_values, dtype=np.float64).ravel()
    pit = pit[np.isfinite(pit)]
    if len(pit) == 0:
        return {"ks_stat": np.nan, "p_value": np.nan, "n": 0}
    ks_stat, p_value = stats.kstest(pit, "uniform", args=(0, 1))
    return {"ks_stat": float(ks_stat), "p_value": float(p_value), "n": int(len(pit))}


def nll_per_sample(
    r_actual: np.ndarray,
    theta_predicted: np.ndarray,
    dt: float,
    n_max: int = 10,
) -> np.ndarray:
    """
    Per-sample NLL for distribution analysis. -log p(r|theta) per formula (8).

    Args:
        r_actual: (T,) observed returns.
        theta_predicted: (T, 5) or (5,) JD params.
        dt: 1 / bars_per_year.
        n_max: truncation.
    Returns:
        (T,) per-sample NLL.
    """
    import torch
    from ..models import jump_diffusion as jd

    r = torch.tensor(r_actual, dtype=torch.float64)
    theta = np.asarray(theta_predicted, dtype=np.float64)
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, (len(r_actual), 5))
    mu = torch.tensor(theta[:, 0], dtype=torch.float64)
    sigma = torch.tensor(theta[:, 1], dtype=torch.float64)
    lam = torch.tensor(theta[:, 2], dtype=torch.float64)
    mu_J = torch.tensor(theta[:, 3], dtype=torch.float64)
    sigma_J = torch.tensor(theta[:, 4], dtype=torch.float64)

    log_p = jd.log_density(r, mu, sigma, lam, mu_J, sigma_J, dt, n_max)
    nll = -log_p.numpy()
    return np.where(np.isfinite(nll), nll, np.nan)
