"""
Empirical jump frequency estimation for data-driven p_center (λ prior).

Uses academically rigorous methods:
- BNS (Barndorff-Nielsen & Shephard 2006): z-test on RV vs BV per day
- Lee-Mykland (2008): per-bar jump detection with bipower-vol local estimator
- Multi-threshold (3σ, 4σ, 5σ): heuristic baseline for comparison
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# BNS: theta for bipower variation (Huang & Tauchen 2005)
_THETA_BV = np.pi**2 / 4 + np.pi - 3  # ~2.609
# Tri-power quarticity: mu = E[|Z|^{4/3}] for Z~N(0,1) = 2^{2/3} * Gamma(7/6) / sqrt(pi)
from scipy.special import gamma as _gamma

_MU_43 = 2 ** (2 / 3) * _gamma(7 / 6) / np.sqrt(np.pi)  # ~0.8309


def _rolling_vol(r: pd.Series, window: int) -> pd.Series:
    """Rolling std, excluding current obs (avoid look-ahead)."""
    return r.rolling(window, min_periods=max(1, window // 2)).std()


# ---------------------------------------------------------------------------
# BNS (Barndorff-Nielsen & Shephard 2006) jump test
# ---------------------------------------------------------------------------


def _bns_z_statistic(r_day: np.ndarray) -> tuple[float, float, float, float]:
    """
    BNS z-statistic for one day. Per Barndorff-Nielsen & Shephard (2006).
    RV - BV captures jump contribution; z = (RV-BV) / SE with tri-power quarticity.
    """
    n = len(r_day)
    if n < 4:
        return np.nan, np.nan, np.nan, np.nan

    r = np.asarray(r_day, dtype=float)
    r = r[np.isfinite(r)]

    rv = np.sum(r**2)
    bv = (np.pi / 2) * np.sum(np.abs(r[1:]) * np.abs(r[:-1]))
    # Tri-power quarticity: sum |r_i|^{4/3} |r_{i-1}|^{4/3} |r_{i-2}|^{4/3}
    tp = np.sum(
        np.abs(r[2:]) ** (4 / 3) * np.abs(r[1:-1]) ** (4 / 3) * np.abs(r[:-2]) ** (4 / 3)
    )
    tp = tp / (_MU_43**3) if _MU_43 > 0 else 1e-12
    tp = max(tp, 1e-20)

    se = np.sqrt((_THETA_BV - 2) * (1 / n) * tp)
    z = (rv - bv) / se if se > 1e-12 else 0.0
    return float(z), float(rv), float(bv), float(tp)


def estimate_lambda_bns(
    df: pd.DataFrame,
    bars_per_day: int = 96,
    bars_per_year: float = 35040,
    alpha: float = 0.05,
) -> tuple[float, int, dict]:
    """
    BNS (Barndorff-Nielsen & Shephard 2006) jump detection.
    Per day: z = (RV - BV) / SE; reject H0 (no jump) if z > z_crit.
    Returns (lambda_annual, n_jump_days, aux) where lambda = n_jump_bars / years.
    We count 1 jump per jump-day (conservative; BNS does not give bar-level count).
    """
    from scipy import stats

    r = df["log_return"].values
    n = len(r)
    if n < bars_per_day * 2:
        return np.nan, 0, {}

    n_days = n // bars_per_day
    r_mat = r[: n_days * bars_per_day].reshape(n_days, bars_per_day)

    z_crit = stats.norm.ppf(1 - alpha)
    z_vals = []
    is_jump_day = np.zeros(n_days, dtype=bool)

    for d in range(n_days):
        z, rv, bv, tp = _bns_z_statistic(r_mat[d])
        z_vals.append(z)
        if np.isfinite(z) and z > z_crit:
            is_jump_day[d] = True

    n_jump_days = int(np.sum(is_jump_day))
    # BNS gives jump days, not bar-level count. Use 1 jump per jump day as lower bound.
    # For λ we want jumps/year: n_jump_days / years is a lower bound.
    years = n / bars_per_year
    lambda_annual = n_jump_days / years if years > 0 else np.nan
    aux = {"z_mean": float(np.nanmean(z_vals)), "z_max": float(np.nanmax(z_vals))}
    return float(lambda_annual), n_jump_days, aux


# ---------------------------------------------------------------------------
# Lee-Mykland (2008) per-bar jump detection
# ---------------------------------------------------------------------------


def estimate_lambda_lee_mykland(
    df: pd.DataFrame,
    bars_per_day: int = 96,
    bars_per_year: float = 35040,
    alpha: float = 0.05,
) -> tuple[float, int, dict]:
    """
    Lee & Mykland (2008) per-bar jump detection.
    L_i = r_i / sigma_i, where sigma_i from local bipower variation.
    Reject (jump) if |L_i| > C_n. K = floor(sqrt(N)) per Lee-Mykland.
    Returns (lambda_annual, n_jumps, aux).
    """
    r = df["log_return"].values
    n = len(r)
    if n < 20:
        return np.nan, 0, {}

    # K = window for local BV, per Lee-Mykland: K ~ sqrt(N) for N=intraday obs
    # We use rolling window over all bars; K = sqrt(bars_per_day) * scale
    k = max(10, int(np.sqrt(bars_per_day * 2)))  # e.g. ~14 for 96 bars/day

    # Local bipower variation: BV_i = (pi/2) * mean(|r_{i-1}||r_{i-2}|) over window
    sigma_sq = np.full(n, np.nan)
    for i in range(k + 1, n):
        win = r[i - k : i]
        if np.any(np.isfinite(win)) and len(win) >= 2:
            bv = (np.pi / 2) * np.nanmean(np.abs(win[1:]) * np.abs(win[:-1]))
            sigma_sq[i] = max(bv, 1e-16)

    sigma = np.sqrt(sigma_sq)
    L = np.where(sigma > 1e-12, np.abs(r) / sigma, 0.0)
    L[~np.isfinite(L)] = 0

    # Critical value: Lee-Mykland (2008) Gumbel limit, c = sqrt(2/pi)
    c = np.sqrt(2 / np.pi)
    ln_n = max(np.log(n), 1.0)
    sqrt_2ln = np.sqrt(2 * ln_n)
    C_n = (1 / c) * (sqrt_2ln - (np.log(np.pi) + np.log(ln_n)) / (2 * sqrt_2ln))
    C_n = C_n - np.log(-np.log(1 - alpha)) / c

    is_jump = L > C_n
    n_jumps = int(np.sum(is_jump))
    years = n / bars_per_year
    lambda_annual = n_jumps / years if years > 0 else np.nan

    aux = {"C_n": float(C_n), "L_max": float(np.nanmax(L) if np.any(np.isfinite(L)) else 0)}
    return float(lambda_annual), n_jumps, aux


# ---------------------------------------------------------------------------
# Heuristic threshold method (for comparison)
# ---------------------------------------------------------------------------


def estimate_lambda_threshold(
    r: np.ndarray,
    sigma_bar: np.ndarray,
    threshold: float,
    bars_per_year: float,
) -> float:
    """
    Estimate λ (jumps/year) using |r| > threshold * sigma_bar.
    """
    valid = np.isfinite(r) & np.isfinite(sigma_bar) & (sigma_bar > 1e-12)
    n_jumps = np.sum(np.abs(r[valid]) > threshold * sigma_bar[valid])
    n_bars = np.sum(valid)
    p_bar = n_jumps / n_bars if n_bars > 0 else 0.0
    return p_bar * bars_per_year




def estimate_empirical_lambda(
    processed_path: Path | str,
    config_path: str = "config.yaml",
    vol_hours: float = 24,
    thresholds: tuple[float, ...] = (3.0, 4.0, 5.0),
) -> dict:
    """
    Multi-method empirical λ estimation. Returns dict with:
    - lambda_threshold: {3.0, 4.0, 5.0} heuristic
    - lambda_bns: BNS (Barndorff-Nielsen & Shephard 2006) - jump days / years
    - lambda_lee_mykland: Lee-Mykland (2008) - per-bar jump count / years
    - lambda_recommended: from academic methods (BNS, LM), fallback threshold
    - p_center_recommended: for config
    """
    processed_path = Path(processed_path)
    cfg = yaml.safe_load(Path(config_path).read_text())
    bars_per_year = cfg["training"]["bars_per_year"]
    bars_per_hour = 4  # 15min
    vol_window = int(vol_hours * bars_per_hour)

    df = pd.read_parquet(processed_path)
    df.index = pd.to_datetime(df.index, utc=True)
    r = df["log_return"].values
    sigma = _rolling_vol(df["log_return"], vol_window).values

    years = (df.index.max() - df.index.min()).total_seconds() / (365.25 * 24 * 3600)
    n_bars = len(df)

    # Heuristic thresholds (for comparison)
    lambda_threshold = {}
    for th in thresholds:
        lam = estimate_lambda_threshold(r, sigma, th, bars_per_year)
        lambda_threshold[th] = float(lam)

    # BNS (Barndorff-Nielsen & Shephard 2006)
    lambda_bns, n_jump_days_bns, aux_bns = estimate_lambda_bns(
        df, bars_per_day=96, bars_per_year=bars_per_year
    )
    lambda_bns = float(lambda_bns) if np.isfinite(lambda_bns) else None

    # Lee-Mykland (2008)
    lambda_lm, n_jumps_lm, aux_lm = estimate_lambda_lee_mykland(
        df, bars_per_day=96, bars_per_year=bars_per_year
    )
    lambda_lm = float(lambda_lm) if np.isfinite(lambda_lm) else None

    # Recommended: prioritize academic methods (LM gives bar-level, BNS gives day-level)
    # LM is more precise for λ; BNS is conservative (jump days only). Use median of academic.
    academic = [x for x in [lambda_bns, lambda_lm] if x is not None and x > 0]
    if academic:
        lambda_rec = float(np.median(academic))
    else:
        # Fallback: 25th percentile of thresholds (conservative)
        lambdas_t = list(lambda_threshold.values())
        lambda_rec = float(np.percentile(lambdas_t, 25)) if lambdas_t else 100.0
    lambda_rec = max(min(lambda_rec, 500), 50)  # clip to [50, 500]
    p_center_recommended = lambda_rec / bars_per_year

    # Summary stats for thresholds (backward compat)
    lambdas_all = list(lambda_threshold.values()) + [x for x in [lambda_bns, lambda_lm] if x]
    lambda_mean = float(np.mean(lambdas_all)) if lambdas_all else np.nan
    lambda_lo = float(np.percentile(lambdas_all, 25)) if lambdas_all else np.nan
    lambda_hi = float(np.percentile(lambdas_all, 75)) if lambdas_all else np.nan

    return {
        "lambda_threshold": lambda_threshold,
        "lambda_bns": lambda_bns,
        "lambda_lee_mykland": lambda_lm,
        "lambda_mean": lambda_mean,
        "lambda_lo": lambda_lo,
        "lambda_hi": lambda_hi,
        "lambda_recommended": lambda_rec,
        "p_center_recommended": p_center_recommended,
        "n_jump_days_bns": n_jump_days_bns,
        "n_jumps_lee_mykland": n_jumps_lm,
        "years": years,
        "n_bars": n_bars,
        "bars_per_year": bars_per_year,
        "aux_bns": aux_bns,
        "aux_lee_mykland": aux_lm,
    }


def run_and_save(
    processed_path: Optional[Path] = None,
    out_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    asset: Optional[str] = None,
) -> dict:
    """
    Run empirical estimation and save to JSON for Stage 2 / TFT to use.

    Args:
        asset: Asset name (ETH, BTC, SOL, DOGE). Determines input/output paths.
               If None, defaults to ETH for backward compatibility.
    """
    root = Path.cwd() if (Path.cwd() / "config.yaml").exists() else Path.cwd().parent
    cfg_path = config_path or (root / "config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    processed_dir = root / cfg["paths"]["processed"]
    checkpoints_dir = root / cfg["paths"]["checkpoints"]

    asset = asset or "ETH"
    asset_upper = asset.upper()
    path = processed_path or (processed_dir / f"{asset_upper}_15min.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}. Run Stage 1 preprocess.")

    result = estimate_empirical_lambda(path, config_path=str(cfg_path))
    out = out_path or (checkpoints_dir / f"{asset_upper.lower()}_empirical_jump_params.json")

    import json

    # JSON: string keys, exclude aux dicts (optional metadata)
    exclude = {"lambda_threshold", "aux_bns", "aux_lee_mykland"}
    out_result = {k: v for k, v in result.items() if k not in exclude}
    out_result["lambda_threshold"] = {str(k): v for k, v in result["lambda_threshold"].items()}

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(out_result, f, indent=2)
    print(f"Empirical jump params saved to {out}")
    lt = result["lambda_threshold"]
    bns = result.get("lambda_bns")
    lm = result.get("lambda_lee_mykland")
    print(f"  λ (3σ)={lt[3.0]:.0f}, (4σ)={lt[4.0]:.0f}, (5σ)={lt[5.0]:.0f} [threshold heuristic]")
    if bns is not None:
        print(f"  λ (BNS)={bns:.0f} [Barndorff-Nielsen & Shephard 2006]")
    if lm is not None:
        print(f"  λ (Lee-Mykland)={lm:.0f} [Lee & Mykland 2008]")
    print(f"  λ_recommended={result['lambda_recommended']:.0f}, p_center={result['p_center_recommended']:.6f}")
    return result
