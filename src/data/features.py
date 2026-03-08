"""
Stage 1 Task 1.4: Build TFT input format from 15min OHLCV + onchain.
Output: data/features/{ASSET}_tft.parquet (feature series) + {ASSET}_tft_arrays.npz (X_hist, Z_future, y)
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config from YAML."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """14-period RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _bb_width(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    """Bollinger Band width: (upper - lower) / middle."""
    middle = close.rolling(period, min_periods=1).mean()
    std = close.rolling(period, min_periods=1).std().replace(0, 1e-10)
    upper = middle + k * std
    lower = middle - k * std
    return (upper - lower) / middle.replace(0, 1e-10)


def _volume_log_change(vol: pd.Series, clip_lo: float = -5.0, clip_hi: float = 5.0) -> pd.Series:
    """log(vol_t / vol_{t-1}), clip to avoid inf, fillna(0)."""
    v = vol.astype(float)
    v_prev = v.shift(1)
    v = v.where(v > 0)
    v_prev = v_prev.where(v_prev > 0)
    out = np.log(v / v_prev)
    out = out.replace([np.inf, -np.inf], np.nan).clip(clip_lo, clip_hi).fillna(0)
    return out


def build_asset_features(
    ohlcv_15m: pd.DataFrame,
    onchain: pd.DataFrame,
    window: int = 96,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Merge OHLCV and onchain, compute all features.
    onchain has hourly index; ffill to 15min.
    """
    df = ohlcv_15m.copy()
    df.index = pd.to_datetime(df.index, utc=True)

    # Align onchain 1h to 15min (ffill within each hour)
    onchain_15m = onchain.reindex(df.index).ffill()

    # Price features: volatility at 1h / 6h / 24h
    df["realized_vol_1h"] = df["log_return"].rolling(4, min_periods=1).std()
    df["realized_vol_6h"] = df["log_return"].rolling(24, min_periods=1).std()
    df["realized_vol_24h"] = df["log_return"].rolling(96, min_periods=1).std()
    df["volume_log_change"] = _volume_log_change(df["volume"])
    df["rsi_14"] = _rsi(df["close"], 14)
    df["bb_width_20"] = _bb_width(df["close"], 20)
    # vol-of-vol (6h window on realized_vol_1h), 24h skew, bar range
    df["vol_of_vol"] = df["realized_vol_1h"].rolling(24, min_periods=1).std()
    df["return_skew_24h"] = df["log_return"].rolling(96, min_periods=1).skew()
    high = df["high"].astype(float).where(df["high"] > 0)
    low = df["low"].astype(float).where(df["low"] > 0)
    hl = np.log(high / low)
    df["high_low_range"] = hl.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Fill remaining NaN from rolling (e.g. skew early bars)
    for col in ["realized_vol_1h", "vol_of_vol", "return_skew_24h"]:
        df[col] = df[col].fillna(0)

    # Onchain (z-scored, from Task 1.2)
    for col in onchain_15m.columns:
        df[col] = onchain_15m[col]

    return df


def _time_covariates(index: pd.DatetimeIndex) -> np.ndarray:
    """hour_sin, hour_cos, dow_sin, dow_cos. Shape (N, 4)."""
    hour = index.hour + index.minute / 60
    dow = index.dayofweek
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7),
        np.cos(2 * np.pi * dow / 7),
    ])


# Feature columns for X_hist (order matters); n_features = 14
_X_FEATURE_COLS = [
    "log_return",
    "realized_vol_1h",
    "realized_vol_6h",
    "realized_vol_24h",
    "volume_log_change",
    "rsi_14",
    "bb_width_20",
    "vol_of_vol",
    "return_skew_24h",
    "high_low_range",
    "active_addresses_z",
    "gas_fee_z",
    "whale_transfers_z",
    "dex_volume_z",
]


def build_tft_arrays(
    df: pd.DataFrame,
    window: int,
    horizon: int,
    asset_id: int,
) -> dict[str, Any]:
    """
    Build TFT format: X_hist (T, window, n_features), Z_future (T, horizon, 4), y (T,).
    Only rows with no NaN in features are kept.
    """
    for col in _X_FEATURE_COLS:
        if col not in df.columns:
            raise KeyError(f"Missing feature column: {col}")

    feats = df[_X_FEATURE_COLS].copy()
    feats = feats.fillna(0)  # fallback for edge NaN
    y = df["log_return"].values
    timestamps = df.index
    split = df["split"].values if "split" in df.columns else np.array(["train"] * len(df))

    # Valid start index: need full window of history
    n = len(feats)
    T = n - window
    if T <= 0:
        return {
            "X_hist": np.zeros((0, window, len(_X_FEATURE_COLS))),
            "Z_future": np.zeros((0, horizon, 4)),
            "y": np.zeros(0),
            "timestamps": pd.DatetimeIndex([]),
            "asset_id": asset_id,
            "split": np.array([]),
        }

    X_hist = np.zeros((T, window, len(_X_FEATURE_COLS)), dtype=np.float32)
    Z_future = np.zeros((T, horizon, 4), dtype=np.float32)
    y_out = np.zeros(T, dtype=np.float32)
    ts_out: list[pd.Timestamp] = []
    split_out: list[str] = []

    feat_arr = feats.values
    z_arr = _time_covariates(timestamps)

    for i in range(window, n):
        t = i - window
        X_hist[t] = feat_arr[i - window : i]
        Z_future[t] = z_arr[i : i + horizon]
        y_out[t] = y[i]
        ts_out.append(timestamps[i])
        split_out.append(split[i] if isinstance(split[i], str) else "train")

    return {
        "X_hist": X_hist,
        "Z_future": Z_future,
        "y": y_out,
        "timestamps": pd.DatetimeIndex(ts_out, tz="UTC"),
        "asset_id": asset_id,
        "split": np.array(split_out),
    }


def run_features(
    config_path: str = "config.yaml",
    assets: Optional[list[str]] = None,
) -> None:
    """
    Build TFT features for each asset, save to data/features/.
    """
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed"])
    onchain_dir = Path(config["paths"]["processed_onchain"])
    feat_dir = Path(config["paths"]["features"])
    feat_dir.mkdir(parents=True, exist_ok=True)

    window = config["stage1"].get("window", 96)
    horizon = config["stage1"].get("horizon", 1)
    asset_list = assets or list(config["stage1"]["assets"].keys())
    asset_ids = {"ETH": 0, "BTC": 1, "SOL": 2, "DOGE": 3}

    for asset in asset_list:
        ohlcv_path = processed_dir / f"{asset}_15min.parquet"
        onchain_path = onchain_dir / f"{asset}_onchain_hourly.parquet"
        missing = []
        if not ohlcv_path.exists():
            missing.append(str(ohlcv_path))
        if not onchain_path.exists():
            missing.append(str(onchain_path))
        if missing:
            print(f"[features] Skip {asset}: missing {missing}")
            continue

        print(f"[features] {asset}...")
        ohlcv = pd.read_parquet(ohlcv_path)
        onchain = pd.read_parquet(onchain_path)
        if onchain.index.name != "hour":
            onchain = onchain.set_index("hour") if "hour" in onchain.columns else onchain

        df = build_asset_features(ohlcv, onchain, window, horizon)
        out_feat = feat_dir / f"{asset}_tft_features.parquet"
        df.to_parquet(out_feat, index=True)
        print(f"  -> {out_feat.name} ({len(df)} rows)")

        arrs = build_tft_arrays(df, window, horizon, asset_ids.get(asset, 0))
        npz_path = feat_dir / f"{asset}_tft_arrays.npz"
        np.savez_compressed(
            npz_path,
            X_hist=arrs["X_hist"],
            Z_future=arrs["Z_future"],
            y=arrs["y"],
            timestamps=arrs["timestamps"].tz_localize(None).values,
            asset_id=np.array(arrs["asset_id"]),
            split=arrs["split"],
        )
        print(f"  -> {npz_path.name} (T={len(arrs['y'])})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--assets", nargs="*")
    args = parser.parse_args()
    run_features(config_path=args.config, assets=args.assets or None)
