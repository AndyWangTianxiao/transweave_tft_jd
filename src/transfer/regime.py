"""
Regime identification via Gaussian HMM (formula 13, doc/stage5_regime.md).
M4: Full-feature PCA pipeline. Fixed n_states=4.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def prepare_hmm_features(
    feature_arrays: Dict[str, np.ndarray],
    missing_threshold: float = 0.5,
    pca_variance_target: float = 0.85,
    pca_max_components: int = 6,
    downsample_factor: int = 4,
) -> Dict[str, Any]:
    """
    M4: Full-feature PCA pipeline. doc/stage5_regime.md Section 5.1.1.
    X_hist (T, window, n_features) -> last timestep -> drop high-missing cols ->
    ffill+mean fill -> Global Scaler -> PCA (>=85% var, max 6) -> downsample.
    """
    assets = list(feature_arrays.keys())
    if not assets:
        return {}

    # Step 1: last timestep
    raw = {a: feature_arrays[a][:, -1, :].astype(np.float64) for a in assets}
    n_features = raw[assets[0]].shape[1]

    # Step 2: missing rate (4 assets pooled)
    all_data = np.concatenate([raw[a] for a in assets], axis=0)
    missing_rates = np.isnan(all_data).mean(axis=0)
    kept_columns = np.where(missing_rates <= missing_threshold)[0]

    # Step 3: fill
    filled = {}
    for a in assets:
        x = raw[a][:, kept_columns].copy()
        df = pd.DataFrame(x)
        df = df.ffill().bfill()
        col_means = df.mean()
        df = df.fillna(col_means)
        filled[a] = df.values.astype(np.float64)

    # Step 4: Global Scaler
    all_filled = np.concatenate([filled[a] for a in assets], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_filled)
    scaled = {a: scaler.transform(filled[a]) for a in assets}

    # Step 5: PCA
    all_scaled = np.concatenate([scaled[a] for a in assets], axis=0)
    n_kept = len(kept_columns)
    pca_full = PCA(n_components=min(pca_max_components, n_kept))
    pca_full.fit(all_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, pca_variance_target) + 1)
    n_components = min(n_components, pca_max_components)
    n_components = max(n_components, 2)

    pca = PCA(n_components=n_components)
    pca.fit(all_scaled)

    # Step 6: Transform + downsample
    hmm_inputs = {}
    for a in assets:
        transformed = pca.transform(scaled[a])
        hmm_inputs[a] = transformed[::downsample_factor]

    return {
        "hmm_inputs": hmm_inputs,
        "pca": pca,
        "scaler": scaler,
        "kept_columns": kept_columns,
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "missing_rates": {int(j): float(missing_rates[j]) for j in range(n_features)},
        "n_dropped_columns": n_features - len(kept_columns),
    }


def _extract_hmm_features(
    X_hist: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    rv_transform: str = "log",
    downsample_rate: int = 4,
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Extract (log_return, 24h_realized_vol) for HMM. Per doc/stage5_regime.md:
    - log_return: from y (or X_hist last bar col 0)
    - 24h RV: X_hist[:, -1, 3] (realized_vol_24h), transform with log(rv+eps) or sqrt(rv)

    Args:
        X_hist: (T, 96, 14) from npz
        y: (T,) log returns
        split: (T,) "train" | "val" | "test"
        rv_transform: "log" or "sqrt"
        downsample_rate: take every Nth sample (15min -> 1h when rate=4)
        max_samples: cap samples for HMM training
    Returns:
        (N, 2) array: [log_return, rv_transformed]
    """
    train_mask = np.asarray(split) == "train"
    r = np.asarray(y[train_mask], dtype=np.float64)
    # X_hist col 3 = realized_vol_24h at last bar of window
    rv = X_hist[train_mask, -1, 3].astype(np.float64)
    rv = np.maximum(rv, 1e-10)
    if rv_transform == "log":
        rv_t = np.log(rv + 1e-10)
    else:
        rv_t = np.sqrt(rv)
    X = np.column_stack([r, rv_t])
    # Drop rows with NaN/Inf (HMM fit rejects non-finite)
    valid = np.isfinite(X).all(axis=1)
    X = X[valid]
    # Downsample
    X = X[::downsample_rate]
    if max_samples is not None and len(X) > max_samples:
        X = X[:max_samples]
    return X


def fit_hmm_regime(
    X: np.ndarray,
    n_iter: int = 100,
    covariance_type: str = "full",
    random_states: Optional[List[int]] = None,
    config_path: str = "config.yaml",
) -> Tuple[GaussianHMM, int]:
    """
    Fit HMM with fixed n_states. Per doc/stage5_regime.md Section 3.5.
    All assets use n_states=4 (config: hmm_n_states_fixed).

    Args:
        X: (N, d) PCA features, already scaled
        n_iter, covariance_type: HMM params
        random_states: multiple seeds, take best loglik
    Returns:
        (best_model, n_states)
    """
    config = _load_config(config_path)
    xfer = config.get("transfer", {})
    n_states = int(xfer.get("hmm_n_states_fixed", 4))
    n_iter = xfer.get("hmm_n_iter", n_iter)
    covariance_type = xfer.get("hmm_covariance_type", covariance_type)
    random_states = xfer.get("hmm_random_states", random_states or [42])

    best_model = None
    best_ll = -np.inf
    for rs in random_states:
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type=covariance_type,
                n_iter=n_iter,
                random_state=rs,
            )
            model.fit(X)
            ll = model.score(X)
            if ll > best_ll:
                best_ll = ll
                best_model = model
        except Exception:
            continue
    if best_model is None:
        raise RuntimeError(f"HMM fit failed for n_states={n_states}")

    return best_model, n_states


def _eigen_decompose(
    P: np.ndarray,
    smooth_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eigen decomposition of transition matrix P. Right eigenvectors, L2 normalized.
    Handles complex (np.real_if_close). Per formula 13.
    """
    P_smooth = P + smooth_eps
    P_smooth = P_smooth / P_smooth.sum(axis=1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eig(P_smooth)
    eigenvalues = np.real_if_close(eigenvalues)
    eigenvectors = np.real_if_close(eigenvectors)
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    for j in range(eigenvectors.shape[1]):
        norm = np.linalg.norm(eigenvectors[:, j]) + 1e-10
        eigenvectors[:, j] = eigenvectors[:, j] / norm
    return eigenvalues, eigenvectors


def fit_regime_all_assets(
    assets: List[str],
    config_path: str = "config.yaml",
) -> Dict[str, Dict[str, Any]]:
    """
    Fit HMM for each asset. M4: uses prepare_hmm_features (full PCA) when
    hmm_feature_mode != "legacy". Saves to experiments/checkpoints/regime_{asset}.npz.
    """
    config = _load_config(config_path)
    root = Path(__file__).resolve().parents[2]
    feat_dir = root / config["paths"]["features"]
    ckpt_dir = root / config["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    xfer = config.get("transfer", {})
    hmm_feature_mode = xfer.get("hmm_feature_mode", "pca")
    smooth_eps = xfer.get("hmm_smooth_eps", 1e-6)

    if hmm_feature_mode == "pca":
        # M4: Full-feature PCA pipeline
        feature_arrays = {}
        for asset in assets:
            npz_path = feat_dir / f"{asset}_tft_arrays.npz"
            if not npz_path.exists():
                print(f"Skipping {asset}: {npz_path} not found")
                continue
            data = np.load(npz_path, allow_pickle=True)
            train_mask = np.asarray(data["split"]) == "train"
            X_hist = data["X_hist"][train_mask]
            if len(X_hist) < 100:
                print(f"Skipping {asset}: {len(X_hist)} train samples (need >= 100)")
                continue
            feature_arrays[asset] = X_hist

        if not feature_arrays:
            return {}

        prep = prepare_hmm_features(
            feature_arrays,
            missing_threshold=xfer.get("hmm_missing_threshold", 0.5),
            pca_variance_target=xfer.get("pca_variance_target", 0.85),
            pca_max_components=xfer.get("pca_max_components", 6),
            downsample_factor=xfer.get("hmm_downsample_factor", 4),
        )
        print(f"PCA: n_components={prep['n_components']}, "
              f"explained_var={prep['cumulative_variance'][-1]:.3f}, "
              f"dropped_cols={prep['n_dropped_columns']}")

        asset_X = prep["hmm_inputs"]
    else:
        # Legacy: 2D return + RV (use hmm_downsample_factor, fallback to rate for compat)
        downsample_rate = xfer.get("hmm_downsample_factor", xfer.get("hmm_downsample_rate", 4))
        max_samples = xfer.get("hmm_max_samples", 24000)
        rv_transform = xfer.get("rv_transform", "log")
        scaler_mode = xfer.get("hmm_scaler_mode", "per_asset")
        asset_X = {}
        for asset in assets:
            npz_path = feat_dir / f"{asset}_tft_arrays.npz"
            if not npz_path.exists():
                print(f"Skipping {asset}: {npz_path} not found")
                continue
            data = np.load(npz_path, allow_pickle=True)
            X = _extract_hmm_features(
                data["X_hist"], data["y"], data["split"],
                rv_transform=rv_transform,
                downsample_rate=downsample_rate,
                max_samples=max_samples,
            )
            if len(X) < 100:
                continue
            asset_X[asset] = X

        if scaler_mode == "global" and asset_X:
            X_all = np.vstack(list(asset_X.values()))
            global_scaler = StandardScaler()
            global_scaler.fit(X_all)
            asset_X = {a: global_scaler.transform(asset_X[a]) for a in asset_X}
        elif scaler_mode == "eth_anchor" and "ETH" in asset_X:
            eth_scaler = StandardScaler()
            eth_scaler.fit(asset_X["ETH"])
            asset_X = {a: eth_scaler.transform(asset_X[a]) for a in asset_X}
        else:
            asset_X = {a: StandardScaler().fit_transform(asset_X[a]) for a in asset_X}

    results = {}
    for asset in assets:
        if asset not in asset_X:
            continue
        X = np.asarray(asset_X[asset], dtype=np.float64)
        valid = np.isfinite(X).all(axis=1)
        if not valid.all():
            X = X[valid]
        if len(X) < 100:
            print(f"Skipping {asset}: {len(X)} samples after non-finite drop")
            continue

        model, n_states = fit_hmm_regime(X, config_path=config_path)
        state_seq = model.predict(X)
        P = model.transmat_
        eigenvalues, eigenvectors = _eigen_decompose(P, smooth_eps=smooth_eps)

        out = {
            "n_states": n_states,
            "state_sequence": state_seq,
            "transition_matrix": P,
            "means": model.means_,
            "covars": model.covars_,
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
        }
        results[asset] = out

        save_path = ckpt_dir / f"regime_{asset}.npz"
        np.savez(
            save_path,
            n_states=n_states,
            state_sequence=state_seq,
            transition_matrix=P,
            means=model.means_,
            covars=model.covars_,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        print(f"Regime {asset}: n_states={n_states}, saved to {save_path}")

    return results


def load_regime_result(asset: str, config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load saved regime result for an asset."""
    config = _load_config(config_path)
    root = Path(__file__).resolve().parents[2]
    ckpt_dir = root / config["paths"]["checkpoints"]
    path = ckpt_dir / f"regime_{asset}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Regime result not found: {path}")
    data = np.load(path, allow_pickle=True)
    out = {
        "n_states": int(data["n_states"]),
        "state_sequence": data["state_sequence"],
        "transition_matrix": data["transition_matrix"],
        "eigenvalues": data["eigenvalues"],
        "eigenvectors": data["eigenvectors"],
    }
    if "means" in data:
        out["means"] = data["means"]
    return out
