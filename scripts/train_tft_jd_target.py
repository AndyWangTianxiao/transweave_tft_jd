"""
Train TFT-JD on target assets (BTC, SOL, DOGE) for Stage 8a comparison.
Mirrors ETH training: base, scratch+W, finetune+W. Per doc/stage8a_audit_present.md §3.5.

Usage:
  python scripts/train_tft_jd_target.py --asset BTC [--mode scratch|finetune] [--use_weakness true|false] [--force]
  # base (no weakness):     --use_weakness false  -> {asset}_tft_jd.ckpt
  # scratch+W:             --use_weakness true --mode scratch -> {asset}_tft_jd_w_scratch.ckpt
  # finetune+W:            --use_weakness true --mode finetune -> {asset}_tft_jd_w_finetune.ckpt

Default: skip if checkpoint exists. Use --force to overwrite.
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tft_dataset import get_asset_splits
from src.models import losses
from src.models.tft_jd import build_tft_jd
from src.behavioral.regularizer import weakness_regularizer


def _load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _compute_r_reg(
    mu: torch.Tensor, sigma: torch.Tensor, lam: torch.Tensor,
    mu_J: torch.Tensor, sigma_J: torch.Tensor,
) -> torch.Tensor:
    """R = mean(||θ_t - θ_{t-1}||²) for consecutive samples in batch."""
    theta = torch.stack([mu, sigma, lam, mu_J, sigma_J], dim=1)
    diff = theta[1:] - theta[:-1]
    return (diff ** 2).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TFT-JD on target asset (mirrors ETH)")
    parser.add_argument("--asset", type=str, required=True, choices=["BTC", "SOL", "DOGE"])
    parser.add_argument("--mode", type=str, default="scratch", choices=["scratch", "finetune"],
                        help="scratch=from scratch, finetune=load base then add weakness")
    parser.add_argument("--use_weakness", type=str, default="false", choices=["true", "false"])
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing checkpoint")
    args = parser.parse_args()

    asset = args.asset
    use_weakness = args.use_weakness == "true"
    mode_finetune = args.mode == "finetune"
    cfg = _load_config()
    tft_cfg = cfg["tft_jd"]
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    seed = train_cfg["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    dt = 1.0 / train_cfg["bars_per_year"]
    n_max = train_cfg["jd_truncation_n"]
    m_crps = train_cfg.get("crps_mc_samples", 200)
    batch_size = tft_cfg["batch_size"]
    lr = tft_cfg["lr"]
    weight_decay = tft_cfg["weight_decay"]
    grad_clip = tft_cfg["grad_clip"]
    nll_phase_ratio = tft_cfg["nll_phase_ratio"]
    alpha_ratio = tft_cfg["loss_alpha_target_ratio"]
    beta = tft_cfg["loss_beta"]
    warmup_epochs = tft_cfg["head_warmup_epochs"]
    patience = tft_cfg["early_stopping_patience"]
    nll_clamp = tft_cfg.get("nll_clamp_sigma", 5.0)
    max_epochs = args.epochs or train_cfg.get("epochs", 100)

    # Lambda prior: use target's static JD params if available
    ckpt_dir = ROOT / paths_cfg.get("checkpoints", "experiments/checkpoints")
    lambda_prior_weight = tft_cfg.get("lambda_prior_weight", 0.0)
    lambda_prior_center = None
    for fname in [f"{asset}_static_jd_params.json", f"{asset.lower()}_static_jd_params.json"]:
        map_path = ckpt_dir / fname
        if map_path.exists():
            import json as _json
            with open(map_path) as _f:
                _map = _json.load(_f)
            lambda_prior_center = float(_map.get("lambda", _map.get("lam", 107.0)))
            break
    if lambda_prior_center is None:
        lambda_prior_center = 107.0
    if lambda_prior_weight > 0:
        log_lam_center = float(np.log(lambda_prior_center + 1e-6))
        print(f"Lambda soft prior: weight={lambda_prior_weight}, center={lambda_prior_center:.1f}")

    # Checkpoint naming (mirrors ETH)
    if use_weakness:
        ckpt_name = f"{asset}_tft_jd_w_{args.mode}.ckpt"
    else:
        ckpt_name = f"{asset}_tft_jd.ckpt"
    ckpt_path = ckpt_dir / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if ckpt_path.exists() and not args.force:
        print(f"Checkpoint exists: {ckpt_path.name}, skip (use --force to overwrite)")
        return

    train_ds, val_ds, test_ds = get_asset_splits(asset)
    n_train = len(train_ds)
    n_segments = max(1, n_train // batch_size)
    segment_starts = np.linspace(0, max(0, n_train - batch_size), n_segments, dtype=int)

    model = build_tft_jd().to(device)
    if mode_finetune and use_weakness:
        base_ckpt = ckpt_dir / f"{asset}_tft_jd.ckpt"
        if base_ckpt.exists():
            ck = torch.load(base_ckpt, map_location=device)
            model.load_state_dict(ck["model_state_dict"], strict=True)
            print(f"Loaded {base_ckpt.name} for finetune with weakness")
        else:
            print(f"WARNING: {base_ckpt.name} not found, starting from scratch")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_nll = float("inf")
    patience_counter = 0
    alpha = 0.0
    train_start = time.time()
    nll_phase_epochs = int(max_epochs * nll_phase_ratio)
    crps_phase_epochs = max_epochs - nll_phase_epochs

    weak_tag = f" + weakness (mode={args.mode})" if use_weakness else ""
    print(f"Training TFT-JD{weak_tag} on {asset} | {max_epochs} epochs, batch_size={batch_size}, n_train={n_train}")
    print(f"Phase 1 (NLL): 1-{nll_phase_epochs} | Phase 2 (+CRPS+R+W): {nll_phase_epochs+1}-{max_epochs}")
    print("-" * 60)

    for epoch in range(max_epochs):
        model.train()
        if epoch < warmup_epochs:
            model.freeze_encoder()
        else:
            model.unfreeze_encoder()

        use_crps = epoch >= nll_phase_epochs
        if use_crps and alpha == 0.0:
            model.eval()
            nll_sum, crps_sum, n_b = 0.0, 0.0, 0
            with torch.no_grad():
                perm = np.random.permutation(len(segment_starts))
                for _ in range(min(20, len(perm))):
                    s = segment_starts[perm[_]]
                    idx = list(range(s, min(s + batch_size, n_train)))
                    if len(idx) < 2:
                        continue
                    xb = torch.stack([train_ds.X_hist[j] for j in idx]).to(device)
                    zb = torch.stack([train_ds.Z_future[j] for j in idx]).to(device)
                    yb = torch.stack([train_ds.y[j] for j in idx]).to(device)
                    mu, sigma, lam, mu_J, sigma_J = model(xb, zb)
                    nll_sum += losses.nll(yb, mu, sigma, lam, mu_J, sigma_J, dt, n_max).item()
                    crps_sum += losses.crps_mc(yb, mu, sigma, lam, mu_J, sigma_J, dt, m_crps, seed, device).item()
                    n_b += 1
            if n_b > 0 and crps_sum > 1e-10:
                mean_nll = nll_sum / n_b
                mean_crps = crps_sum / n_b
                alpha = alpha_ratio * (mean_nll / mean_crps)
                print(f"[Phase 2] Dynamic alpha = {alpha:.6f}")
            model.train()

        perm = np.random.permutation(len(segment_starts))
        epoch_nll, epoch_crps, epoch_r, epoch_weak, epoch_lam_prior, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        epoch_start = time.time()
        pbar = tqdm(range(len(perm)), desc=f"Epoch {epoch+1}/{max_epochs}", leave=False, disable=len(perm) < 10)

        for i in pbar:
            s = segment_starts[perm[i]]
            idx = list(range(s, min(s + batch_size, n_train)))
            if len(idx) < 2:
                continue

            xb = torch.stack([train_ds.X_hist[j] for j in idx]).to(device)
            zb = torch.stack([train_ds.Z_future[j] for j in idx]).to(device)
            yb = torch.stack([train_ds.y[j] for j in idx]).to(device)

            mu, sigma, lam, mu_J, sigma_J = model(xb, zb)

            nll_val = losses.nll_clamped(yb, mu, sigma, lam, mu_J, sigma_J, dt, n_max, clamp_sigma=nll_clamp)
            r_reg = _compute_r_reg(mu, sigma, lam, mu_J, sigma_J) if use_crps and beta > 0 else torch.tensor(0.0, device=device)

            if lambda_prior_weight > 0:
                l_lam_prior = lambda_prior_weight * ((torch.log(lam + 1e-6) - log_lam_center) ** 2).mean()
            else:
                l_lam_prior = torch.tensor(0.0, device=device)

            if use_crps:
                crps_val = losses.crps_mc(yb, mu, sigma, lam, mu_J, sigma_J, dt, m_crps, seed + epoch, device)
                loss = nll_val + alpha * crps_val + beta * r_reg + l_lam_prior
                epoch_crps += crps_val.item()
                epoch_r += r_reg.item()
            else:
                loss = nll_val + l_lam_prior

            epoch_lam_prior += l_lam_prior.item()
            if use_weakness:
                l_weak = weakness_regularizer(mu, sigma, lam, mu_J, sigma_J, dt, cvar_method="analytic")
                loss = loss + l_weak
                epoch_weak += l_weak.item()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            epoch_nll += nll_val.item()
            n_batches += 1

        if n_batches == 0:
            continue

        mean_nll = epoch_nll / n_batches
        mean_crps = epoch_crps / n_batches if use_crps else 0.0

        model.eval()
        val_nll_sum, val_n = 0.0, 0
        with torch.no_grad():
            for j in range(0, len(val_ds), batch_size):
                idx = list(range(j, min(j + batch_size, len(val_ds))))
                xb = torch.stack([val_ds.X_hist[k] for k in idx]).to(device)
                zb = torch.stack([val_ds.Z_future[k] for k in idx]).to(device)
                yb = torch.stack([val_ds.y[k] for k in idx]).to(device)
                mu, sigma, lam, mu_J, sigma_J = model(xb, zb)
                vnll = losses.nll(yb, mu, sigma, lam, mu_J, sigma_J, dt, n_max)
                val_nll_sum += vnll.item()
                val_n += yb.numel()
        val_nll = val_nll_sum / max(val_n, 1)

        phase_tag = "NLL" if not use_crps else ("NLL+CRPS+R+W" if use_weakness else "NLL+CRPS+R")
        print(f"Epoch {epoch+1:3d}/{max_epochs} [{phase_tag}] train_NLL={mean_nll:.4f} val_NLL={val_nll:.4f} ({time.time()-epoch_start:.1f}s)")

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_nll": val_nll,
                "asset": asset,
            }, ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - train_start
    print("-" * 60)
    print(f"Done. Best val NLL: {best_val_nll:.4f} | Elapsed: {elapsed/60:.1f} min | Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
