"""
Feature space shift D_H per paper Definition 3.1: Z^(x) = H^(x) × Θ^(x).
Doc/stage5_regime.md Section 10 Priority 4: compute distribution difference between
source and target assets in H (feature) space.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def _load_asset_features(
    asset: str,
    split: str = "train",
    config_path: str = "config.yaml",
) -> np.ndarray:
    """
    Load X_hist for asset, train split only. Flatten to (T, n_features).
    Returns last bar of each window (most recent) for simplicity, or mean over window.
    """
    config = _load_config(config_path)
    root = Path(__file__).resolve().parents[2]
    feat_dir = root / config["paths"]["features"]
    npz_path = feat_dir / f"{asset}_tft_arrays.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Features not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X_hist = data["X_hist"]  # (T, window, n_features)
    split_arr = np.asarray(data["split"])
    mask = split_arr == split
    X = X_hist[mask]  # (N, window, n_features)
    # Use last bar of window (most recent)
    X = X[:, -1, :].astype(np.float64)
    return X


def _psi_bins(
    p: np.ndarray,
    q: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI) between two distributions.
    PSI = sum_i (q_i - p_i) * ln(q_i / p_i). Bins from quantiles of p.
    """
    eps = 1e-10
    p = np.asarray(p).ravel()
    q = np.asarray(q).ravel()
    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]
    if len(p) < 10 or len(q) < 10:
        return 0.0
    edges = np.percentile(p, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    hist_p, _ = np.histogram(p, bins=edges)
    hist_q, _ = np.histogram(q, bins=edges)
    hist_p = hist_p / (hist_p.sum() + eps) + eps
    hist_q = hist_q / (hist_q.sum() + eps) + eps
    psi = np.sum((hist_q - hist_p) * np.log(hist_q / hist_p))
    return float(psi)


def _js_divergence(p_hist: np.ndarray, q_hist: np.ndarray, eps: float = 1e-10) -> float:
    """Jensen-Shannon divergence between two histograms (normalized)."""
    p = p_hist / (p_hist.sum() + eps) + eps
    q = q_hist / (q_hist.sum() + eps) + eps
    m = (p + q) / 2
    js = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    return float(js)


def _missing_rate(X: np.ndarray, zero_as_missing: bool = True) -> float:
    """Fraction of zeros (or NaN) in X. For onchain ffill, 0 often means missing."""
    if zero_as_missing:
        return float(np.mean(X == 0))
    return float(np.mean(~np.isfinite(X)))


def compute_feature_shift(
    asset_a: str,
    asset_b: str,
    split: str = "train",
    n_bins: int = 10,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """
    Feature space shift D_H per Definition 3.1 (paper_desc.md).
    Computes distribution difference between assets in H (feature) space.

    Args:
        asset_a: source asset (e.g. ETH)
        asset_b: target asset (e.g. BTC, SOL, DOGE)
        split: "train" | "val" | "test"
        n_bins: for PSI / histogram
    Returns:
        D_H: aggregate feature shift (mean PSI over features)
        psi_per_feature: PSI per feature column
        missing_rate_a, missing_rate_b: fraction of zeros (proxy for ffill)
        missing_penalty: exp(-|missing_a - missing_b| / 0.2) for S' adjustment
    """
    X_a = _load_asset_features(asset_a, split=split, config_path=config_path)
    X_b = _load_asset_features(asset_b, split=split, config_path=config_path)
    n_feat = X_a.shape[1]
    # Align lengths (min)
    n = min(len(X_a), len(X_b))
    X_a = X_a[:n]
    X_b = X_b[:n]

    psi_list = []
    for j in range(n_feat):
        psi = _psi_bins(X_a[:, j], X_b[:, j], n_bins=n_bins)
        psi_list.append(psi)
    D_H = float(np.mean(psi_list))
    psi_per_feature = np.array(psi_list)

    missing_a = _missing_rate(X_a)
    missing_b = _missing_rate(X_b)
    missing_penalty = np.exp(-abs(missing_a - missing_b) / 0.2)

    return {
        "D_H": D_H,
        "psi_per_feature": psi_per_feature,
        "missing_rate_a": missing_a,
        "missing_rate_b": missing_b,
        "missing_penalty": missing_penalty,
    }


def compute_s_transfer_adjusted(
    s_transfer: float,
    D_H: float,
    missing_penalty: float,
    w_h: Optional[float] = None,
    config_path: str = "config.yaml",
) -> float:
    """
    S' = S · exp(-D_H / W_H) · missing_penalty.
    Diagnostic: D_H large or missing penalty low → S' lower.
    w_h from config transfer.d_h_w_h (default 1.0, larger = less aggressive).
    """
    if w_h is None:
        w_h = _load_config(config_path).get("transfer", {}).get("d_h_w_h", 1.0)
    return s_transfer * np.exp(-D_H / w_h) * missing_penalty
