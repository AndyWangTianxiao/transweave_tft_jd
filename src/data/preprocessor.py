"""
Stage 1 Task 1.3: Aggregate 1min OHLCV to multi-frequency, log returns, jump detection.
Output: data/processed/{ASSET}_{freq}.parquet
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config from YAML."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def _resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
) -> pd.DataFrame:
    """Resample 1min OHLCV: Open=first, High=max, Low=min, Close=last, Volume=sum."""
    ts_col = "timestamp" if "timestamp" in df.columns else df.index.name
    if ts_col and ts_col in df.columns:
        df = df.set_index(ts_col)
    def _first(x):
        return x.iloc[0] if len(x) > 0 else np.nan

    def _last(x):
        return x.iloc[-1] if len(x) > 0 else np.nan

    return df.resample(freq).agg({
        "open": _first,
        "high": "max",
        "low": "min",
        "close": _last,
        "volume": "sum",
    }).dropna(how="all")


def _log_return(close: pd.Series) -> pd.Series:
    """Log return: r_t = log(close_t / close_{t-1})."""
    return np.log(close / close.shift(1))


def _forward_fill_limit(series: pd.Series, limit: int = 5) -> pd.Series:
    """Forward fill NaN, max `limit` consecutive bars. Rest stay NaN."""
    return series.ffill(limit=limit)


def _rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    """Rolling std of returns."""
    return returns.rolling(window=window, min_periods=1).std()


def process_asset(
    asset: str,
    raw_path: Path,
    out_dir: Path,
    freqs: list[str],
    train_end: str,
    val_end: str,
    test_end: str,
    jump_sigma: float,
    jump_vol_hours: float,
    ffill_limit: int = 5,
) -> None:
    """
    Load 1min OHLCV, resample to freqs, add log_return, jump, split.
    """
    df = pd.read_parquet(raw_path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    # Map user-friendly freq to pandas resample rule (use "min" not deprecated "T")
    _FREQ_MAP = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "1h"}
    for freq in freqs:
        rule = _FREQ_MAP.get(freq, freq)
        resampled = _resample_ohlcv(df.copy(), rule)

        # Log return
        resampled["log_return"] = _log_return(resampled["close"])

        # Forward fill (max 5 bars)
        resampled["log_return"] = _forward_fill_limit(
            resampled["log_return"], limit=ffill_limit
        )

        # Jump: bars per hour from freq
        if freq == "1min":
            bars_per_hour = 60
        elif freq == "5min" or freq == "5T":
            bars_per_hour = 12
        elif freq == "15min" or freq == "15T":
            bars_per_hour = 4
        elif freq == "1h" or freq == "1H":
            bars_per_hour = 1
        else:
            bars_per_hour = 60  # default
        vol_window = int(jump_vol_hours * bars_per_hour)
        sigma = _rolling_vol(resampled["log_return"], vol_window)
        resampled["is_jump"] = (
            resampled["log_return"].abs() > jump_sigma * sigma.replace(0, 1e-10)
        )

        # Time split
        train_end_ts = pd.Timestamp(train_end, tz="UTC")
        val_end_ts = pd.Timestamp(val_end, tz="UTC")
        test_end_ts = pd.Timestamp(test_end, tz="UTC")

        def _split(t):
            if t < train_end_ts:
                return "train"
            if t < val_end_ts:
                return "val"
            return "test"

        resampled["split"] = resampled.index.map(_split)

        # Filename: 5T->5min, 15T->15min, 1h->1h
        fname = freq.replace("T", "min").replace("H", "h").lower()
        out_path = out_dir / f"{asset}_{fname}.parquet"
        resampled.to_parquet(out_path, index=True)
        print(f"  -> {out_path.name} ({len(resampled)} rows)")


def run_preprocess(
    config_path: str = "config.yaml",
    assets: Optional[list[str]] = None,
    freqs: Optional[list[str]] = None,
) -> None:
    """
    Preprocess 1min OHLCV: resample, log return, jump, split.
    """
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_ohlcv"])
    out_dir = Path(config["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1 = config["stage1"]
    train_end = stage1["train_end"] + " 23:59:59"
    val_end = stage1["val_end"] + " 23:59:59"
    test_end = stage1["test_end"] + " 23:59:59"
    jump_sigma = stage1.get("jump_sigma_threshold", 3)
    jump_vol_hours = stage1.get("jump_vol_window_hours", 24)

    asset_list = assets or list(stage1["assets"].keys())
    freq_list = freqs or ["5min", "15min", "1h"]

    for asset in asset_list:
        raw_path = raw_dir / f"{asset}_1min.parquet"
        if not raw_path.exists():
            print(f"[preprocess] Skip {asset}: {raw_path} not found")
            continue
        print(f"[preprocess] {asset}...")
        process_asset(
            asset,
            raw_path,
            out_dir,
            freq_list,
            train_end,
            val_end,
            test_end,
            jump_sigma,
            jump_vol_hours,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--assets", nargs="*")
    parser.add_argument("--freqs", nargs="*", default=["5min", "15min", "1h"])
    args = parser.parse_args()
    run_preprocess(
        config_path=args.config,
        assets=args.assets or None,
        freqs=args.freqs,
    )
