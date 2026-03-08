"""
Stage 6: TransWeave transfer experiment. Per doc/stage6_transfer.md.
Requires: eth_tft_jd_w_finetune.ckpt, stage5_transfer_report.json, regime_ETH.npz, {ASSET}_tft_arrays.npz.
Outputs: weak_model_{BTC,SOL,DOGE}.pt, transfer_map_{BTC,SOL,DOGE}.pt,
         stage6_experiment_results.json, stage6_comparison_table.csv.
Usage: python scripts/run_stage6.py
Config: reads config.yaml from project root by default.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tft_dataset import get_split_indices, load_feature_arrays
from src.models.tft_jd import build_tft_jd, infer_eth_theta
from src.transfer.transweave import run_transfer_experiment


def _to_json_serializable(obj):
    """Convert numpy types for json."""
    if isinstance(obj, (np.bool_, np.integer)):
        return bool(obj) if isinstance(obj, np.bool_) else int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    return obj


def main() -> None:
    config_path = "config.yaml"
    cfg = yaml.safe_load(open(ROOT / config_path))
    ckpt_dir = ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ETH TFT-JD
    eth_ckpt = ckpt_dir / "eth_tft_jd_w_finetune.ckpt"
    if not eth_ckpt.exists():
        raise FileNotFoundError(
            f"ETH checkpoint not found: {eth_ckpt}. Run scripts/train_tft_jd_eth.py first."
        )
    model = build_tft_jd()
    ck = torch.load(eth_ckpt, map_location=device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.to(device)

    # Load ETH features and infer theta
    X_hist, Z_future, _, split_arr = load_feature_arrays("ETH", config_path)
    eth_split_indices = get_split_indices(split_arr)
    print(f"ETH split: train_end={eth_split_indices['train_end']}, val_end={eth_split_indices['val_end']}")

    print("Inferring ETH theta (full sequence)...")
    theta_eth_15min = infer_eth_theta(
        model, X_hist, Z_future,
        batch_size=cfg.get("transfer_map", {}).get("batch_size", 2048),
        device=device,
    )
    print(f"  theta_eth length: {len(theta_eth_15min)}")

    # Load Stage 5 report
    stage5_path = ckpt_dir / "stage5_transfer_report.json"
    if not stage5_path.exists():
        raise FileNotFoundError(
            f"Stage 5 report not found: {stage5_path}. Run scripts/run_stage5.py first."
        )
    with open(stage5_path) as f:
        stage5_report = json.load(f)
    assert stage5_report.get("version") == "v3_paper_formula", "Stage 5 report version mismatch"

    # Run experiments for each target
    all_results = []
    for target in ["BTC", "SOL", "DOGE"]:
        try:
            results = run_transfer_experiment(
                target_asset=target,
                stage5_report=stage5_report,
                theta_eth_15min=theta_eth_15min,
                eth_split_indices=eth_split_indices,
                config=cfg,
                config_path=config_path,
                ckpt_dir=ckpt_dir,
            )
            all_results.append(results)
        except Exception as e:
            print(f"[{target}] Failed: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    out_json = ckpt_dir / "stage6_experiment_results.json"
    serializable = [_to_json_serializable(r) for r in all_results]
    with open(out_json, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Build comparison table
    rows = []
    for r in all_results:
        target = r["target_asset"]
        decision = r["decision"]
        row = {"target": target, "decision": decision}
        if "scratch" in r:
            row["scratch_nll"] = r["scratch"]["nll"]
            row["scratch_crps"] = r["scratch"]["crps"]
        if "direct" in r:
            row["direct_nll"] = r["direct"]["nll"]
            row["direct_crps"] = r["direct"]["crps"]
        if "transweave" in r:
            row["transweave_nll"] = r["transweave"]["nll"]
            row["transweave_crps"] = r["transweave"]["crps"]
        if "force_transfer" in r:
            row["force_transfer_nll"] = r["force_transfer"]["nll"]
            row["force_transfer_crps"] = r["force_transfer"]["crps"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = ckpt_dir / "stage6_comparison_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"Comparison table saved to {out_csv}")
    print("\n" + df.to_string())


if __name__ == "__main__":
    main()
