"""
Stage 1 Task 1.2: Load raw on-chain CSVs, merge by asset, rolling z-score.
Output: data/processed/onchain/{ASSET}_onchain_hourly.parquet
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config from YAML. Path relative to project root."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


# Mapping: asset -> list of (csv_file, target_column_in_df)
# Final schema: active_addresses, gas_fee, whale_transfers, dex_volume_smart
_ASSET_CSV_MAP = {
    "ETH": [
        ("eth_active_addresses_hourly.csv", "active_addresses", None),
        ("eth_avg_gas_gwei_hourly.csv", "gas_fee", "avg_gas_gwei"),
        ("eth_whale_transfers_hourly.csv", "whale_transfers", None),
        ("eth_dex_volume_smart_hourly.csv", "dex_volume_smart", None),
    ],
    "BTC": [
        ("btc_active_addresses_hourly.csv", "active_addresses", None),
        (None, "gas_fee", 0),  # no gas, fill 0
        ("btc_whale_transfers_hourly.csv", "whale_transfers", None),
        (None, "dex_volume_smart", 0),  # no dex, fill 0
    ],
    "SOL": [
        ("sol_tx_fee_hourly.csv", "active_addresses", "tx_count"),  # tx_count as proxy
        ("sol_tx_fee_hourly.csv", "gas_fee", "avg_fee_sol"),
        ("sol_whale_dex_hourly.csv", "whale_transfers", None),
        ("sol_whale_dex_hourly.csv", "dex_volume_smart", None),
    ],
    "DOGE": [
        ("doge_active_addresses_hourly.csv", "active_addresses", None),
        (None, "gas_fee", 0),
        ("doge_whale_transfers_hourly.csv", "whale_transfers", None),
        ("doge_dex_volume_smart_hourly.csv", "dex_volume_smart", None),
    ],
}


def _parse_hour(s: str) -> pd.Timestamp:
    """Parse hour string to UTC-aware timestamp."""
    return pd.to_datetime(s, utc=True)


def _load_csv(path: Path, hour_col: str = "hour") -> pd.DataFrame:
    """Load CSV and ensure hour is datetime index."""
    df = pd.read_csv(path)
    df[hour_col] = pd.to_datetime(df[hour_col], utc=True)
    return df


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - mean) / std over window. No future leak."""
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    return (series - mean) / std.replace(0, 1)  # avoid div by zero


def build_asset_onchain(
    asset: str,
    raw_dir: Path,
    start_hour: str,
    end_hour: str,
    zscore_window: int,
) -> pd.DataFrame:
    """
    Build on-chain DataFrame for one asset: merge CSVs, fill missing with 0, z-score.
    Output columns: active_addresses_z, gas_fee_z, whale_transfers_z, dex_volume_smart_z.
    """
    hour_range = pd.date_range(
        start=start_hour,
        end=end_hour,
        freq="h",
        inclusive="left",
        tz="UTC",
    )
    result = pd.DataFrame(index=hour_range)
    result.index.name = "hour"

    for item in _ASSET_CSV_MAP[asset]:
        csv_file, target_col, source_col = item
        if csv_file is None:
            result[target_col] = item[2]  # constant fill
            continue

        path = raw_dir / csv_file
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        df = _load_csv(path)

        if source_col is None:
            source_col = target_col
        vals = df.set_index("hour")[source_col]
        # Dune exports <nil> for null; coerce to numeric, fill with 0
        vals = pd.to_numeric(vals, errors="coerce").fillna(0)

        # Reindex to full hour range, fill missing with 0
        merged = vals.reindex(hour_range, fill_value=0)
        result[target_col] = merged

    # Rolling z-score (no future leak)
    for col in result.columns:
        result[f"{col}_z"] = _rolling_zscore(result[col], zscore_window)

    out_cols = [c for c in result.columns if c.endswith("_z")]
    out = result[out_cols].copy()
    if "dex_volume_smart_z" in out.columns:
        out = out.rename(columns={"dex_volume_smart_z": "dex_volume_z"})
    return out


def run_onchain(
    config_path: str = "config.yaml",
    assets: Optional[list[str]] = None,
) -> None:
    """
    Process raw on-chain CSVs: merge by asset, z-score, save to processed/onchain/.
    """
    config = load_config(config_path)
    raw_dir = Path(config["paths"]["raw_onchain"])
    out_dir = Path(config["paths"]["processed_onchain"])
    out_dir.mkdir(parents=True, exist_ok=True)

    zscore_window = config.get("stage1", {}).get("onchain_zscore_window", 168)
    start = config["stage1"]["start_date"] + " 00:00:00"
    end = "2025-07-01 00:00:00"  # exclusive end

    asset_list = assets or list(config["stage1"]["assets"].keys())

    for asset in asset_list:
        print(f"[onchain] {asset}...")
        df = build_asset_onchain(
            asset, raw_dir, start, end, zscore_window
        )
        out_path = out_dir / f"{asset}_onchain_hourly.parquet"
        df.to_parquet(out_path, index=True)
        print(f"  -> {out_path.name} ({len(df)} rows)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--assets", nargs="*", help="Assets to process (default: all)")
    args = parser.parse_args()
    run_onchain(config_path=args.config, assets=args.assets or None)
