"""
TFT dataset: load ETH_tft_arrays.npz, filter by split, return (X_hist, Z_future, y).
Shape consistent with npz: X_hist (N, 96, 14), Z_future (N, 1, 4), y (N,).
"""

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import yaml


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load config from project root."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


class TFTDataset(Dataset):
    """
    Dataset for TFT-JD: (X_hist, Z_future, y) per sample.
    Loads from {asset}_tft_arrays.npz, filters by split.
    """

    def __init__(
        self,
        asset: str = "ETH",
        split: Literal["train", "val", "test"] = "train",
        data_dir: Optional[Path] = None,
        config_path: str = "config.yaml",
    ) -> None:
        """
        Args:
            asset: Asset name (e.g. ETH, BTC).
            split: train, val, or test.
            data_dir: Override features dir; if None, use config paths.features.
            config_path: Config file path.
        """
        super().__init__()
        cfg = _load_config(config_path)
        root = Path(__file__).resolve().parents[2]
        feat_dir = data_dir or (root / cfg.get("paths", {}).get("features", "data/features"))
        npz_path = feat_dir / f"{asset}_tft_arrays.npz"

        data = np.load(npz_path, allow_pickle=True)
        X_hist = data["X_hist"]  # (T, 96, 14)
        Z_future = data["Z_future"]  # (T, 1, 4)
        y = data["y"]  # (T,)
        split_arr = np.asarray(data["split"])  # (T,) object array of "train"/"val"/"test"

        # Drop rows with NaN/Inf in y (would cause NLL=nan)
        valid = np.isfinite(y)
        if not valid.all():
            import warnings
            n_drop = int((~valid).sum())
            warnings.warn(f"Dropping {n_drop} samples with non-finite y in {asset} {split}")
            X_hist = X_hist[valid]
            Z_future = Z_future[valid]
            y = y[valid]
            split_arr = split_arr[valid]

        mask = split_arr == split
        self.X_hist = torch.from_numpy(X_hist[mask].astype(np.float32))
        self.Z_future = torch.from_numpy(Z_future[mask].astype(np.float32))
        self.y = torch.from_numpy(y[mask].astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_hist[idx], self.Z_future[idx], self.y[idx]


def get_eth_splits(
    config_path: str = "config.yaml",
) -> Tuple[TFTDataset, TFTDataset, TFTDataset]:
    """
    Return (train, val, test) TFTDataset for ETH.
    """
    return get_asset_splits("ETH", config_path)


def get_asset_splits(
    asset: str,
    config_path: str = "config.yaml",
    data_dir: Optional[Path] = None,
) -> Tuple[TFTDataset, TFTDataset, TFTDataset]:
    """
    Return (train, val, test) TFTDataset for any asset.
    """
    cfg = _load_config(config_path)
    root = Path(__file__).resolve().parents[2]
    feat_dir = data_dir or (root / cfg.get("paths", {}).get("features", "data/features"))

    train_ds = TFTDataset(asset=asset, split="train", data_dir=feat_dir, config_path=config_path)
    val_ds = TFTDataset(asset=asset, split="val", data_dir=feat_dir, config_path=config_path)
    test_ds = TFTDataset(asset=asset, split="test", data_dir=feat_dir, config_path=config_path)
    return train_ds, val_ds, test_ds


def get_split_indices(split_arr: np.ndarray) -> Dict[str, int]:
    """
    Return train_end, val_end indices for stage6. npz stores [train, val, test] in order.
    Per doc/stage6_transfer.md Section 1.4.
    """
    split_arr = np.asarray(split_arr)
    train_end = int((split_arr == "train").sum())
    val_end = train_end + int((split_arr == "val").sum())
    return {"train_end": train_end, "val_end": val_end, "total": len(split_arr)}


def load_feature_arrays(
    asset: str,
    config_path: str = "config.yaml",
    data_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load X_hist, Z_future, y, split_arr from npz for stage6.
    Returns raw arrays (no filtering by split).
    """
    cfg = _load_config(config_path)
    root = Path(__file__).resolve().parents[2]
    feat_dir = data_dir or (root / cfg.get("paths", {}).get("features", "data/features"))
    npz_path = feat_dir / f"{asset}_tft_arrays.npz"
    data = np.load(npz_path, allow_pickle=True)
    X_hist = data["X_hist"]
    Z_future = data["Z_future"]
    y = data["y"].astype(np.float64)
    split_arr = np.asarray(data["split"])
    return X_hist, Z_future, y, split_arr


def load_returns(
    asset: str,
    split: Literal["train", "val", "test"],
    config_path: str = "config.yaml",
    data_dir: Optional[Path] = None,
) -> np.ndarray:
    """Load y (log returns) for given split. Per doc/stage6_transfer.md Section 1.3."""
    _, _, y, split_arr = load_feature_arrays(asset, config_path, data_dir)
    mask = split_arr == split
    return y[mask]
