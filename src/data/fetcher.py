"""
Stage 1 Task 1.1: OHLCV data fetcher via ccxt.
Fetch 1min OHLCV for ETH, BTC, SOL, DOGE.
Supports binance, kraken, okx, kucoin (use kraken/okx/kucoin if Binance returns HTTP 451).
"""

from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import requests
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config from YAML. Path relative to project root."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def _get_output_path(asset: str, config: dict) -> Path:
    """Return path for raw OHLCV parquet."""
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / config["paths"]["raw_ohlcv"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{asset}_1min.parquet"


def _parse_ohlcv_candle(row: list) -> dict:
    """
    Parse ccxt OHLCV row: [timestamp_ms, open, high, low, close, volume].
    Returns dict with timestamp (UTC naive) and OHLCV.
    """
    ts_ms = int(row[0])
    return {
        "timestamp": pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_localize(None),
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
    }


def fetch_ohlcv(
    symbol: str,
    since: pd.Timestamp,
    until: pd.Timestamp,
    exchange: ccxt.Exchange,
    timeframe: str = "1m",
    limit: int = 1000,
    sleep_seconds: float = 1.0,
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Fetch OHLCV in batches. ccxt returns up to ~1000 candles per request.
    Retries on timeout. Reports progress.
    """
    import time

    since_ms = int(since.timestamp() * 1000)
    until_ms = int(until.timestamp() * 1000)
    total_minutes = (until_ms - since_ms) / (1000 * 60)
    est_batches = max(1, int(total_minutes / (limit or 1)))

    all_rows: list[dict] = []
    current = since_ms
    batch_idx = 0

    while current < until_ms:
        for attempt in range(max_retries):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current, limit=limit)
                break
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 451:
                    raise RuntimeError(
                        "HTTP 451: Binance blocks your region. "
                        "In config.yaml set stage1.fetcher.exchange to 'kraken', 'okx', or 'kucoin'."
                    ) from e
                raise
            except Exception as e:
                err_name = type(e).__name__
                if "Timeout" in err_name or "timeout" in str(e).lower():
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        print(f"       [retry {attempt+1}/{max_retries}] Timeout, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
                else:
                    raise

        if not ohlcv:
            break

        for row in ohlcv:
            parsed = _parse_ohlcv_candle(row)
            if parsed["timestamp"] >= until:
                break
            all_rows.append(parsed)

        batch_idx += 1
        pct = min(100, batch_idx / est_batches * 100)
        latest = pd.Timestamp(ohlcv[-1][0], unit="ms") if ohlcv else since
        print(f"       batch {batch_idx}: {len(all_rows)} rows | ~{pct:.1f}% | latest {latest}")

        last_ts = ohlcv[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1
        time.sleep(sleep_seconds)

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] < until]
    return df


def run_fetch(
    config_path: str = "config.yaml",
    assets: Optional[list[str]] = None,
) -> None:
    """
    Main entry: fetch 1min OHLCV for configured assets with resume support.
    If parquet exists, continues from latest timestamp.
    """
    config = load_config(config_path)
    stage1 = config["stage1"]
    cfg = stage1["fetcher"]

    symbols = assets or list(stage1["assets"].values())
    start = pd.Timestamp(stage1["start_date"], tz=None)
    end = pd.Timestamp(stage1["end_date"], tz=None) + pd.Timedelta(days=1)

    ex_name = cfg["exchange"].lower()
    if ex_name not in ("binance", "kraken", "okx", "kucoin"):
        raise ValueError(
            f"Exchange '{ex_name}' not supported. Use binance, kraken, okx, or kucoin."
        )
    timeout_ms = cfg.get("timeout_ms", 60000)
    max_retries = cfg.get("max_retries", 5)
    exchange = getattr(ccxt, ex_name)(
        {"enableRateLimit": True, "timeout": timeout_ms}
    )
    timeframe = cfg["timeframe"]
    limit = cfg["limit_per_request"]
    sleep_sec = cfg["sleep_seconds"]

    print(f"Exchange: {ex_name} | timeout: {timeout_ms}ms | retries: {max_retries}")

    for symbol in symbols:
        asset = symbol.split("/")[0]
        out_path = _get_output_path(asset, config)

        since = start
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            if not existing.empty:
                last_ts = pd.Timestamp(existing["timestamp"].max())
                since = last_ts + pd.Timedelta(minutes=1)
                print(f"[{asset}] Resuming from {since} (existing {len(existing)} rows)")
                if since >= end:
                    print(f"[{asset}] Already complete. Skip.")
                    continue

        dfs_to_concat = []
        if out_path.exists():
            dfs_to_concat.append(pd.read_parquet(out_path))

        print(f"[{asset}] Fetching {symbol} from {since} to {end}")
        new_df = fetch_ohlcv(
            symbol, since, end, exchange, timeframe, limit, sleep_sec, max_retries
        )
        if not new_df.empty:
            dfs_to_concat.append(new_df)

        if dfs_to_concat:
            combined = pd.concat(dfs_to_concat, ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"])
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            combined.to_parquet(out_path, index=False)
            print(f"[{asset}] Saved {len(combined)} rows to {out_path}")
            print(f"       Range: {combined['timestamp'].min()} -- {combined['timestamp'].max()}")
        else:
            print(f"[{asset}] No new data.")


if __name__ == "__main__":
    run_fetch()
