"""
Tail risk evaluation: VaR backtest, CVaR accuracy.
Per doc/stage8a_audit_present.md Task 2.2.
"""

from typing import Union

import numpy as np
from scipy import stats

from .calibration import _jd_cdf


def _jd_var(
    alpha: float,
    mu: float,
    sigma: float,
    lam: float,
    mu_J: float,
    sigma_J: float,
    dt: float,
    n_max: int = 10,
) -> float:
    """VaR_alpha: quantile of JD distribution. Solve F(VaR) = alpha via bisection."""
    # Rough bounds: normal approx gives mu*dt - 2*sigma*sqrt(dt) to mu*dt + 2*sigma*sqrt(dt)
    drift = mu * dt
    vol = sigma * np.sqrt(dt)
    lo = drift - 5 * vol
    hi = drift + 5 * vol
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        f_mid = _jd_cdf(np.array([mid]), mu, sigma, lam, mu_J, sigma_J, dt, n_max)[0]
        if abs(f_mid - alpha) < 1e-8:
            return mid
        if f_mid < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def var_backtest(
    r_actual: np.ndarray,
    theta_predicted: np.ndarray,
    alpha: float = 0.05,
    dt: float = 1.0 / 35040,
    n_max: int = 10,
) -> dict:
    """
    VaR exceedance: breach_rate, expected_rate, kupiec_pvalue.

    Args:
        r_actual: (T,) observed returns.
        theta_predicted: (T, 5) JD params per step.
        alpha: VaR confidence level (e.g. 0.05 for 5%).
        dt: 1 / bars_per_year.
        n_max: JD truncation.
    Returns:
        {"breach_rate": float, "expected_rate": float, "n_breaches": int,
         "n_total": int, "kupiec_pvalue": float}
    """
    r = np.asarray(r_actual, dtype=np.float64).ravel()
    theta = np.asarray(theta_predicted, dtype=np.float64)
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, (len(r), 5))

    n = len(r)
    if n == 0:
        return {"breach_rate": np.nan, "expected_rate": alpha, "n_breaches": 0, "n_total": 0, "kupiec_pvalue": np.nan}

    # VaR at each step (left tail: losses)
    var_vals = np.zeros(n)
    for i in range(n):
        var_vals[i] = _jd_var(
            alpha,
            theta[i, 0], theta[i, 1], theta[i, 2], theta[i, 3], theta[i, 4],
            dt, n_max,
        )
    breaches = r < var_vals
    n_breaches = int(breaches.sum())
    breach_rate = n_breaches / n

    # Kupiec (1995) LR test: H0 breach_rate = alpha
    if n_breaches == 0 or n_breaches == n:
        kupiec_pvalue = 1e-10  # degenerate
    else:
        log_lr = (
            n_breaches * np.log(breach_rate)
            + (n - n_breaches) * np.log(1 - breach_rate)
            - n_breaches * np.log(alpha)
            - (n - n_breaches) * np.log(1 - alpha)
        )
        lr_stat = -2 * log_lr
        kupiec_pvalue = float(1 - stats.chi2.cdf(lr_stat, 1))

    return {
        "breach_rate": float(breach_rate),
        "expected_rate": float(alpha),
        "n_breaches": n_breaches,
        "n_total": n,
        "kupiec_pvalue": float(kupiec_pvalue),
    }


def cvar_accuracy(
    r_actual: np.ndarray,
    theta_predicted: np.ndarray,
    alpha: float = 0.05,
    dt: float = 1.0 / 35040,
    n_max: int = 10,
    n_samples: int = 1000,
) -> dict:
    """
    Tail loss magnitude accuracy when VaR breached.
    CVaR_alpha = E[r | r <= VaR_alpha]. Compare predicted CVaR vs actual mean of breaches.

    Args:
        r_actual: (T,) observed returns.
        theta_predicted: (T, 5) JD params.
        alpha: confidence level.
        dt, n_max: JD params.
        n_samples: Monte Carlo samples for CVaR estimate per step.
    Returns:
        {"cvar_predicted_mean": float, "cvar_actual_mean": float, "mae": float,
         "n_breaches": int, "breach_mask_used": bool}
    """
    r = np.asarray(r_actual, dtype=np.float64).ravel()
    theta = np.asarray(theta_predicted, dtype=np.float64)
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, (len(r), 5))

    # Get VaR and breach mask
    var_vals = np.array([
        _jd_var(alpha, theta[i, 0], theta[i, 1], theta[i, 2], theta[i, 3], theta[i, 4], dt, n_max)
        for i in range(len(r))
    ])
    breach_mask = r < var_vals
    n_breaches = int(breach_mask.sum())

    if n_breaches < 2:
        return {
            "cvar_predicted_mean": np.nan,
            "cvar_actual_mean": np.nan,
            "mae": np.nan,
            "n_breaches": n_breaches,
            "breach_mask_used": True,
        }

    # Actual CVaR: mean of r when r < VaR
    cvar_actual = float(np.mean(r[breach_mask]))

    # Predicted CVaR: Monte Carlo per breach step, then average
    np.random.seed(42)
    from ..models import jump_diffusion as jd
    import torch

    cvar_pred_list = []
    breach_inds = np.where(breach_mask)[0]
    for idx in breach_inds:
        mu, sigma, lam, mu_J, sigma_J = theta[idx, 0], theta[idx, 1], theta[idx, 2], theta[idx, 3], theta[idx, 4]
        samples = jd.sample_one_step(
            float(mu), float(sigma), float(lam), float(mu_J), float(sigma_J),
            dt, n_samples, 42 + idx, "cpu",
        )
        samples = samples.numpy()
        var_i = var_vals[idx]
        tail = samples[samples <= var_i]
        if len(tail) > 0:
            cvar_pred_list.append(np.mean(tail))
        else:
            cvar_pred_list.append(var_i)
    cvar_pred_mean = float(np.mean(cvar_pred_list))

    return {
        "cvar_predicted_mean": cvar_pred_mean,
        "cvar_actual_mean": cvar_actual,
        "mae": float(np.abs(cvar_pred_mean - cvar_actual)),
        "n_breaches": n_breaches,
        "breach_mask_used": True,
    }
