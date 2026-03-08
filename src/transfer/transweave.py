"""
TransWeave transfer learning: SimpleJDModel, TransferMap, Phase 3/4.
Per doc/stage6_transfer.md. Transfer occurs in JD parameter space, not network weights.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from ..behavioral.weakness import compute_w_pt
from ..data.tft_dataset import get_split_indices, load_feature_arrays, load_returns
from ..models import jump_diffusion as jd
from ..models.losses import crps_mc, nll, nll_clamped


def _load_config(config_path: str = "config.yaml") -> dict:
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


def upsample_regime_to_15min(
    regime_seq_1h: np.ndarray,
    target_len: int,
    downsample_factor: int = 4,
) -> np.ndarray:
    """
    Upsample 1h regime sequence to 15min. Per doc/stage6_transfer.md Section 3.2.
    """
    regime_15min = np.repeat(regime_seq_1h, downsample_factor)
    if len(regime_15min) < target_len:
        regime_15min = np.pad(
            regime_15min, (0, target_len - len(regime_15min)), mode="edge"
        )
    return regime_15min[:target_len].astype(np.int64)


class SimpleJDModel(nn.Module):
    """
    Weak target model: Linear(14→5) with JD param constraints.
    Per doc/stage3_tft.md Task 3.0 and verify_tft_jd.ipynb Linear baseline.
    Same as LinearJDHead: single linear layer, MAP bias init, NLL training.
    """

    def __init__(
        self,
        n_features: int = 14,
        n_jd_params: int = 5,
        hidden_dim: int = 32,
        mu_j_scale: float = 0.06,
        softplus_eps: float = 1e-4,
        lambda_max: float = 35040,
        map_path: Optional[str] = None,
        config_path: str = "config.yaml",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_jd_params)
        self.mu_j_scale = mu_j_scale
        self.softplus_eps = softplus_eps
        self.lambda_max = lambda_max
        # MAP bias init per Stage 3 LinearJDHead
        root = Path(__file__).resolve().parents[2]
        cfg = _load_config(config_path)
        ckpt_dir = root / cfg.get("paths", {}).get("checkpoints", "experiments/checkpoints")
        map_path = map_path or str(ckpt_dir / "eth_static_jd_params.json")
        if Path(map_path).exists():
            import json
            with open(map_path) as f:
                m = json.load(f)
            self._init_from_map(m)
        else:
            nn.init.normal_(self.linear.weight, 0, 1e-3)
            nn.init.zeros_(self.linear.bias)

    def _init_from_map(self, m: dict) -> None:
        """MAP bias init. Per verify_tft_jd.ipynb LinearJDHead._init_from_map."""
        mu = m["mu"]
        sig = m["sigma"]
        lam = m.get("lam", m.get("lambda", 70.0))
        muJ = m.get("mu_J", m.get("mu_j", 0.0))
        sigJ = m.get("sigma_J", m.get("sigma_j", 0.02))
        eps = self.softplus_eps
        b = self.linear.bias.data
        b[0] = mu
        b[1] = float(np.log(np.expm1(np.clip(sig - eps, 1e-6, None))))
        b[2] = float(np.log(np.expm1(np.clip(lam - eps, 1e-6, None))))
        b[3] = float(np.arctanh(np.clip(muJ / self.mu_j_scale, -0.999, 0.999)))
        b[4] = float(np.log(np.expm1(np.clip(sigJ - eps, 1e-6, None))))
        nn.init.normal_(self.linear.weight, 0, 1e-3)

    def forward(self, x_last: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_last: (batch, 14) current bar features.
        Returns:
            theta: (batch, 5) [mu, sigma, lam, mu_j, sigma_j] annualized.
        """
        out = self.linear(x_last)
        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + self.softplus_eps
        lam = (F.softplus(out[:, 2]) + self.softplus_eps).clamp(max=self.lambda_max)
        mu_j = torch.tanh(out[:, 3]) * self.mu_j_scale
        sigma_j = F.softplus(out[:, 4]) + self.softplus_eps
        return torch.stack([mu, sigma, lam, mu_j, sigma_j], dim=-1)


class TransferMap(nn.Module):
    """
    Regime-conditioned MLP: (theta_a, regime_onehot) -> theta_b.
    Per doc/stage6_transfer.md Section 3.1. n_regimes from ETH HMM.

    B2 fix: input/output param-space normalization so MLP works in ~N(0,1) space.
    Buffers a_mean/a_std/b_mean/b_std are set by set_norm_stats() before training.
    If not set (legacy ckpt), falls back to unnormalized residual (backward compatible).
    """

    def __init__(
        self,
        n_jd_params: int = 5,
        n_regimes: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        input_dim = n_jd_params + n_regimes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_jd_params),
        )
        self.residual_scale = nn.Parameter(torch.ones(n_jd_params) * 0.3)
        self.n_jd_params = n_jd_params
        self.n_regimes = n_regimes
        # B2: normalization buffers (set via set_norm_stats before training)
        self.register_buffer("a_mean", torch.zeros(n_jd_params))
        self.register_buffer("a_std", torch.ones(n_jd_params))
        self.register_buffer("b_mean", torch.zeros(n_jd_params))
        self.register_buffer("b_std", torch.ones(n_jd_params))
        self._has_norm = False

    def set_norm_stats(
        self,
        theta_a: torch.Tensor,
        theta_b: torch.Tensor,
    ) -> None:
        """Compute and store per-param mean/std from training data. Call before training."""
        self.a_mean = theta_a.mean(dim=0).detach()
        self.a_std = theta_a.std(dim=0).clamp(min=1e-6).detach()
        self.b_mean = theta_b.mean(dim=0).detach()
        self.b_std = theta_b.std(dim=0).clamp(min=1e-6).detach()
        self._has_norm = True
        print(f"  [TransferMap] norm stats set: a_mean={self.a_mean.tolist()}, b_mean={self.b_mean.tolist()}")

    def forward(
        self,
        theta_a: torch.Tensor,
        regime_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            theta_a: (batch, 5) source JD params [mu, sigma, lam, mu_j, sigma_j]
            regime_onehot: (batch, n_regimes)
        Returns:
            theta_b: (batch, 5) predicted target JD params
        """
        # B2: normalize input to ~N(0,1)
        theta_a_norm = (theta_a - self.a_mean) / self.a_std  # (batch, 5)
        x = torch.cat([theta_a_norm, regime_onehot], dim=-1)
        delta = self.net(x)
        # Residual in normalized space, then denormalize to θ^b space
        theta_b_norm = theta_a_norm + self.residual_scale * delta
        theta_b = self.b_mean + self.b_std * theta_b_norm
        # All modes: T outputs all 5 params. Full/Partial/Weak differ in λ_TW weight,
        # not in which parameters T maps (see final_fix_stage6.md Section 4).
        theta_b = torch.stack([
            theta_b[:, 0],                 # mu
            F.softplus(theta_b[:, 1]),     # sigma > 0
            F.softplus(theta_b[:, 2]),     # lam > 0
            theta_b[:, 3],                 # mu_j
            F.softplus(theta_b[:, 4]),     # sigma_j > 0
        ], dim=-1)
        return theta_b


def train_weak_target_model(
    asset: str,
    config: dict,
    config_path: str = "config.yaml",
) -> SimpleJDModel:
    """
    Train weak target model for 15min theta^(b) sequence. Per doc/stage6_transfer.md Section 2.2.
    """
    X_hist, _, y, split_arr = load_feature_arrays(asset, config_path)
    train_mask = split_arr == "train"
    val_mask = split_arr == "val"

    X_train = X_hist[train_mask, -1, :].astype(np.float64)
    y_train_arr = y[train_mask]
    valid_train = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train_arr)
    X_train = X_train[valid_train]
    y_train_arr = y_train_arr[valid_train]
    if len(X_train) < 100:
        raise ValueError(f"[{asset}] Too few valid train samples ({len(X_train)}) after NaN/Inf filter")
    X_last_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train_arr)

    X_val = X_hist[val_mask, -1, :].astype(np.float64)
    y_val_arr = y[val_mask]
    valid_val = np.isfinite(X_val).all(axis=1) & np.isfinite(y_val_arr)
    X_val = X_val[valid_val]
    y_val_arr = y_val_arr[valid_val]
    X_last_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val_arr)
    if len(X_last_val) < 10:
        X_last_val = X_last_train[-1000:]
        y_val = y_train[-1000:]

    dt = 1.0 / config["training"]["bars_per_year"]
    n_max = config["training"].get("jd_truncation_n", 10)
    tft_cfg = config.get("tft_jd", {})
    tmap_cfg = config.get("transfer_map", {})
    mu_j_scale = tft_cfg.get("mu_j_scale", 0.06)
    softplus_eps = tft_cfg.get("softplus_eps", 1e-4)
    lambda_max = tft_cfg.get("lambda_max", 35040)

    # MAP prior config: reuse static_jd_map params, with a per-sample weight
    prior_cfg = config.get("weak_model_prior", config.get("static_jd_map", {}))
    p_center = float(prior_cfg.get("p_center", 0.002))
    scale_p = float(prior_cfg.get("scale_p", 1.0))
    kappa_center = float(prior_cfg.get("kappa_center", 10.0))
    scale_kappa = float(prior_cfg.get("scale_kappa", 0.5))
    prior_weight = float(prior_cfg.get("prior_weight", 0.1))

    model = SimpleJDModel(
        n_features=X_last_train.shape[1],
        mu_j_scale=mu_j_scale,
        softplus_eps=softplus_eps,
        lambda_max=lambda_max,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=tmap_cfg.get("weak_model_lr", 1e-3))
    batch_size = tmap_cfg.get("batch_size", 2048)
    patience = tmap_cfg.get("weak_model_patience", 10)
    max_epochs = tmap_cfg.get("weak_model_epochs", 100)

    dataset = torch.utils.data.TensorDataset(X_last_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_nll = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            theta = model(x_batch)
            log_p = jd.log_density(
                y_batch, theta[:, 0], theta[:, 1], theta[:, 2],
                theta[:, 3], theta[:, 4], dt, n_max,
            )
            # MAP prior: prevents sigma too small / lambda too large (Stage 2 Scheme 2 analog)
            prior_pen = _map_prior_penalty(theta, dt, p_center, scale_p, kappa_center, scale_kappa)
            loss = -torch.mean(log_p) + prior_weight * prior_pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            theta_val = model(X_last_val)
            val_log_p = jd.log_density(
                y_val, theta_val[:, 0], theta_val[:, 1], theta_val[:, 2],
                theta_val[:, 3], theta_val[:, 4], dt, n_max,
            )
            val_nll = -torch.mean(val_log_p).item()

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        if epoch % 20 == 0:
            print(f"  Weak model [{asset}] epoch {epoch}: val_nll={val_nll:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Weak model [{asset}] done: best val_nll={best_val_nll:.4f}")
    return model


def infer_weak_theta(
    model: SimpleJDModel,
    asset: str,
    config_path: str = "config.yaml",
    split: str = "train",
) -> List[Dict[str, float]]:
    """Infer theta^(b) for target asset train split. Per doc/stage6_transfer.md Section 2.3."""
    X_hist, _, _, split_arr = load_feature_arrays(asset, config_path)
    mask = split_arr == split
    X_last = torch.FloatTensor(X_hist[mask, -1, :])

    model.eval()
    with torch.no_grad():
        theta = model(X_last)

    theta_list = []
    for i in range(theta.shape[0]):
        theta_list.append({
            "mu": theta[i, 0].item(),
            "sigma": theta[i, 1].item(),
            "lam": theta[i, 2].item(),
            "mu_j": theta[i, 3].item(),
            "sigma_j": theta[i, 4].item(),
        })
    return theta_list


def train_transfer_map(
    theta_a_15min: List[Dict[str, float]],
    theta_b_15min: List[Dict[str, float]],
    r_target_train: np.ndarray,
    regime_seq_eth_1h: np.ndarray,
    n_regimes_eth: int,
    config: dict,
    config_path: str = "config.yaml",
    theta_a_val: Optional[List[Dict[str, float]]] = None,
    theta_b_val: Optional[List[Dict[str, float]]] = None,
    r_target_val: Optional[np.ndarray] = None,
    transfer_mode: str = "full",
) -> Tuple[TransferMap, List[Dict[str, float]]]:
    """
    Phase 3: Learn transfer map T. Per TASK_FIX_PHASE4.md Section 2.2 and formula 47.
    L_unified = L_JD + lambda_tw * L_TransWeave + lambda_wpt * L_weakness.

    L_JD         = nll_clamped(r^(b), T(theta^(a)))  (primary term, weight=1)
    L_TransWeave = L_intertwine + lambda_ent * Ent(T)
                 = MSE(T(theta^(a)), theta^(b)_weak) + lambda_ent * Ent_isometry(T)
    L_weakness   = -log W_PT(T(theta^(a))) + MSE(W_PT(T(theta^(a))), W_PT(theta^(b)_weak))

    Full/Partial/Weak differ in lambda_tw (TransWeave weight), not in which params T maps.
    """
    tmap_cfg = config.get("transfer_map", {})
    hidden_dim = tmap_cfg.get("hidden_dim", 64)
    lr = tmap_cfg.get("phase3_lr", 1e-3)
    n_epochs = tmap_cfg.get("phase3_epochs", 50)
    lambda_ent = tmap_cfg.get("lambda_ent", 0.01)
    lambda_wpt = tmap_cfg.get("lambda_wpt", 0.02)
    # Per-mode lambda_tw: Partial/Weak use lower weight (weaker BD intertwine constraint)
    default_lambda_tw = tmap_cfg.get("lambda_tw", config.get("unified", {}).get("lambda_tw", 0.1))
    lambda_tw = tmap_cfg.get(f"{transfer_mode}_lambda_tw", default_lambda_tw)
    phase4_lambda_wpt = tmap_cfg.get("phase4_lambda_wpt", lambda_wpt)
    batch_size = tmap_cfg.get("batch_size", 2048)
    clamp_sigma = tmap_cfg.get(
        "phase4_clamp_sigma",
        config.get("tft_jd", {}).get("nll_clamp_sigma", 5.0),
    )
    lam_max = config.get("tft_jd", {}).get("lambda_max", 35040.0)
    dt = 1.0 / config["training"]["bars_per_year"]
    n_max = config["training"].get("jd_truncation_n", 10)
    loss_alpha_crps = tmap_cfg.get("loss_alpha_crps", config.get("training", {}).get("loss_alpha", 1.0))
    crps_mc_samples = config.get("training", {}).get("crps_mc_samples", 200)
    downsample_factor = config.get("transfer", {}).get("hmm_downsample_factor", 4)
    # Extract W_PT params once to avoid repeated config file reads inside the training loop
    risk_cfg = config.get("risk", {})
    weakness_cfg = config.get("weakness", {})
    wpt_gamma = float(risk_cfg.get("gamma", 7.0))
    wpt_alpha = float(weakness_cfg.get("cvar_alpha", risk_cfg.get("cvar_alpha", 0.05)))

    n = min(len(theta_a_15min), len(theta_b_15min), len(r_target_train))
    n_val = len(theta_a_val) if (theta_a_val is not None) else 0
    regime_15min = upsample_regime_to_15min(
        regime_seq_eth_1h, n + n_val, downsample_factor
    )

    theta_a_t = torch.FloatTensor([
        [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
         t.get("sigma_j", t.get("sigma_J", 0))]
        for t in theta_a_15min[:n]
    ])
    theta_b_t = torch.FloatTensor([
        [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
         t.get("sigma_j", t.get("sigma_J", 0))]
        for t in theta_b_15min[:n]
    ])
    regime_oh = F.one_hot(torch.LongTensor(regime_15min[:n]), n_regimes_eth).float()
    r_train_t = torch.FloatTensor(r_target_train[:n])
    if not torch.isfinite(r_train_t).all():
        r_train_t = torch.nan_to_num(r_train_t, nan=0.0, posinf=0.0, neginf=0.0)

    T = TransferMap(n_jd_params=5, n_regimes=n_regimes_eth, hidden_dim=hidden_dim)
    # B2: set normalization stats from training data
    T.set_norm_stats(theta_a_t, theta_b_t)
    optimizer = torch.optim.Adam(T.parameters(), lr=lr)

    # Same TensorDataset order as Phase 4: (theta_a, theta_b, r, regime) for consistency
    dataset = torch.utils.data.TensorDataset(theta_a_t, theta_b_t, r_train_t, regime_oh)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Per-param scale for MSE normalization: prevents λ dimension from dominating L_TransWeave.
    # Computed from θ^b train data (per TOTAL_FIX Section 2.2 Scheme 2).
    theta_b_scale = theta_b_t.std(dim=0).clamp(min=1e-6).detach()  # (5,)
    print(f"  [Phase 3] θ^b per-param std (for MSE norm): {theta_b_scale.tolist()}")
    # W_PT eps for joint normalization (A1)
    w_pt_eps = float(weakness_cfg.get("w_pt_eps", 1e-8))

    # Val data setup
    has_val = theta_a_val is not None and theta_b_val is not None and len(theta_a_val) > 0
    r_val_t: Optional[torch.Tensor] = None
    if has_val:
        theta_a_val_t = torch.FloatTensor([
            [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
             t.get("sigma_j", t.get("sigma_J", 0))]
            for t in theta_a_val
        ])
        theta_b_val_t = torch.FloatTensor([
            [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
             t.get("sigma_j", t.get("sigma_J", 0))]
            for t in theta_b_val
        ])
        n_val_use = min(len(theta_a_val_t), len(theta_b_val_t), len(regime_15min) - n)
        regime_val_oh = F.one_hot(
            torch.LongTensor(regime_15min[n : n + n_val_use]), n_regimes_eth
        ).float()
        theta_a_val_t = theta_a_val_t[:n_val_use]
        theta_b_val_t = theta_b_val_t[:n_val_use]
        # Val returns for L_JD on val (needed for val_unified)
        if r_target_val is not None:
            r_val_t = torch.FloatTensor(r_target_val[:n_val_use])
            if not torch.isfinite(r_val_t).all():
                r_val_t = torch.nan_to_num(r_val_t, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        theta_a_val_t = theta_b_val_t = regime_val_oh = None

    patience = tmap_cfg.get("phase3_patience", 10)
    print(f"  Phase 3 [{transfer_mode}] lr={lr} λ_wpt={lambda_wpt} λ_tw={lambda_tw} patience={patience}")
    best_val_unified = float("inf")
    best_state = None
    patience_counter = 0
    phase3_history: List[Dict[str, float]] = []

    for epoch in range(n_epochs):
        T.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (theta_a_batch, theta_b_batch, r_batch, regime_batch) in enumerate(loader):
            optimizer.zero_grad()
            b_pred = T(theta_a_batch, regime_batch)
            b_pred = _clamp_theta_for_jd(b_pred, lam_max)

            # L_JD = nll_clamped + alpha*CRPS (formula 48)
            loss_nll = _jd_nll_clamped(r_batch, b_pred, dt, n_max, clamp_sigma)
            loss_crps = _jd_crps(r_batch, b_pred, dt, m_samples=crps_mc_samples)
            loss_jd = loss_nll + loss_alpha_crps * loss_crps

            # L_TransWeave = normalized_MSE + lambda_ent * Ent(T)  (formula 49)
            # Per-param normalized MSE: divide by θ^b std to balance dimensions
            loss_bd = (((b_pred - theta_b_batch) / theta_b_scale) ** 2).mean()
            loss_entropy = _ent_isometry(theta_a_batch, b_pred, collapse_ratio=0.5)
            loss_transweave = loss_bd + lambda_ent * loss_entropy

            # L_weakness (formula 50) — A1: joint normalization
            wpt_pred_raw = _w_pt_from_theta(b_pred, dt, config_path, return_raw=True,
                                             gamma=wpt_gamma, alpha=wpt_alpha)
            wpt_target_raw = _w_pt_from_theta(theta_b_batch, dt, config_path, return_raw=True,
                                               gamma=wpt_gamma, alpha=wpt_alpha)
            joint_max = torch.cat([wpt_pred_raw, wpt_target_raw]).max().detach() + w_pt_eps
            wpt_pred_n = (wpt_pred_raw / joint_max).clamp(min=1e-8, max=1.0)
            wpt_target_n = (wpt_target_raw / joint_max).clamp(min=1e-8, max=1.0)
            loss_wpt_neglog = -torch.log(wpt_pred_n).mean()
            loss_wpt_consist = F.mse_loss(wpt_pred_n, wpt_target_n.detach())
            loss_weakness = loss_wpt_neglog + loss_wpt_consist

            # Loss decomposition logging on first batch
            if epoch == 0 and batch_idx == 0:
                s_jd = loss_jd.item()
                s_tw = loss_transweave.item()
                s_wk_raw = loss_weakness.item()
                s_wk = lambda_wpt * s_wk_raw
                print(f"  [Phase 3] epoch 0 batch 0 loss decomposition:")
                print(f"    loss_JD={s_jd:.4f} (nll+α·crps)  loss_tw={s_tw:.4f}  loss_weak={s_wk_raw:.4f}")
                print(f"    L_unified={s_jd + lambda_tw*s_tw + s_wk:.4f}  batch={len(r_batch)}")

            loss = loss_jd + lambda_tw * loss_transweave + lambda_wpt * loss_weakness

            loss.backward()
            grad_has_nan = any(torch.isnan(p.grad).any().item() for p in T.parameters() if p.grad is not None)
            if not grad_has_nan:
                torch.nn.utils.clip_grad_norm_(T.parameters(), 1.0)
                optimizer.step()
            else:
                optimizer.zero_grad()
            epoch_loss += loss.item() if torch.isfinite(loss).all().item() else 0.0
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Val unified for early stop (TOTAL_FIX Section 3)
        val_unified = None
        val_mse_diag = None
        val_nll_diag = None
        if has_val and theta_a_val_t is not None and regime_val_oh is not None:
            T.eval()
            with torch.no_grad():
                b_val_pred = T(theta_a_val_t, regime_val_oh)
                b_val_pred = _clamp_theta_for_jd(b_val_pred, lam_max)

                # L_JD on val (NLL + CRPS)
                if r_val_t is not None:
                    val_nll_raw = _jd_nll_clamped(
                        r_val_t[:len(b_val_pred)], b_val_pred, dt, n_max, clamp_sigma
                    ).item()
                    val_crps_raw = _jd_crps(
                        r_val_t[:len(b_val_pred)], b_val_pred, dt, m_samples=crps_mc_samples
                    ).item()
                    val_nll_diag = val_nll_raw + loss_alpha_crps * val_crps_raw
                else:
                    val_nll_diag = 0.0  # fallback: no val returns available

                # L_TransWeave on val (normalized)
                val_mse_diag = (((b_val_pred - theta_b_val_t) / theta_b_scale) ** 2).mean().item()
                val_ent = _ent_isometry(theta_a_val_t, b_val_pred, collapse_ratio=0.5).item()
                val_tw = val_mse_diag + lambda_ent * val_ent

                # L_weakness on val — A1: joint normalization
                wpt_val_raw = _w_pt_from_theta(b_val_pred, dt, config_path, return_raw=True,
                                                gamma=wpt_gamma, alpha=wpt_alpha)
                wpt_val_target_raw = _w_pt_from_theta(theta_b_val_t, dt, config_path, return_raw=True,
                                                       gamma=wpt_gamma, alpha=wpt_alpha)
                val_joint_max = torch.cat([wpt_val_raw, wpt_val_target_raw]).max().detach() + w_pt_eps
                wpt_val_n = (wpt_val_raw / val_joint_max).clamp(min=1e-8, max=1.0)
                wpt_val_target_n = (wpt_val_target_raw / val_joint_max).clamp(min=1e-8, max=1.0)
                val_wk_neglog = -torch.log(wpt_val_n).mean().item()
                val_wk_consist = F.mse_loss(wpt_val_n, wpt_val_target_n).item()
                val_wk = val_wk_neglog + val_wk_consist

                # Full val_unified for early stop (L_unified: L_JD + λ_tw·tw + λ_wpt·weak)
                val_unified = val_nll_diag + lambda_tw * val_tw + lambda_wpt * val_wk
                val_JD = val_nll_diag  # L_JD = NLL + α·CRPS
                phase3_history.append({
                    "epoch": float(epoch),
                    "L_train": float(epoch_loss),
                    "val_unified": float(val_unified),
                    "val_JD": float(val_JD),
                    "val_tw": float(val_tw),
                    "val_weak": float(val_wk),
                })
            T.train()

            if val_unified < best_val_unified:
                best_val_unified = val_unified
                best_state = {k: v.clone() for k, v in T.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Phase 3 early stop at epoch {epoch} (val_unified={val_unified:.4f})")
                    break
        else:
            if epoch_loss < best_val_unified:
                best_val_unified = epoch_loss
                best_state = {k: v.clone() for k, v in T.state_dict().items()}
            phase3_history.append({
                "epoch": float(epoch),
                "L_train": float(epoch_loss),
                "val_unified": None,
                "val_JD": None,
                "val_tw": None,
                "val_weak": None,
            })

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            msg = f"  Phase 3 epoch {epoch}: L_train={epoch_loss:.4f}"
            if val_unified is not None:
                val_JD = val_nll_diag
                msg += f" val_unified={val_unified:.4f} (val_JD={val_JD:.4f} val_tw={val_tw:.4f} val_weak={val_wk:.4f})"
            print(msg)

    if best_state is not None:
        T.load_state_dict(best_state)
    return T, phase3_history


def _ent_isometry(
    a_batch: torch.Tensor,
    b_pred: torch.Tensor,
    collapse_ratio: float = 0.5,
) -> torch.Tensor:
    """
    Isometry-based Ent(T) approximation. Per TASK_FIX_PHASE4.md Section 3.1.
    Penalizes when output distance << input distance (collapse); does not encourage divergence.
    """
    n = len(a_batch)
    half = n // 2
    if half < 2:
        return torch.tensor(0.0, device=a_batch.device, dtype=a_batch.dtype)

    d_in = torch.norm(a_batch[:half] - a_batch[half : 2 * half], dim=1)
    d_out = torch.norm(b_pred[:half] - b_pred[half : 2 * half], dim=1)
    ratio = d_out / (d_in + 1e-8)
    loss = torch.relu(collapse_ratio - ratio).mean()
    return loss


def _clamp_theta_for_jd(theta: torch.Tensor, lam_max: float = 35040.0) -> torch.Tensor:
    """
    Clamp theta to valid JD params; replace NaN/Inf. Prevents Poisson/NLL errors.
    Uses out-of-place ops only so gradients flow through in Phase 4 training.
    """
    sigma = theta[:, 1].clamp(min=1e-6, max=10.0)
    lam = torch.where(
        torch.isfinite(theta[:, 2]) & (theta[:, 2] >= 0),
        theta[:, 2].clamp(max=lam_max),
        torch.full_like(theta[:, 2], 1e-6),
    )
    sigma_j = theta[:, 4].clamp(min=1e-6, max=1.0)
    out = torch.stack([theta[:, 0], sigma, lam, theta[:, 3], sigma_j], dim=-1)
    safe = torch.tensor(
        [[0.0, 1e-6, 1e-6, 0.0, 1e-6]], device=theta.device, dtype=theta.dtype
    ).expand_as(out)
    return torch.where(torch.isfinite(out), out, safe)


def _map_prior_penalty(
    theta: torch.Tensor,
    dt: float,
    p_center: float,
    scale_p: float,
    kappa_center: float,
    scale_kappa: float,
) -> torch.Tensor:
    """
    MAP prior penalty (mean over batch). Mirrors fit_static_mle Scheme 2.
    Prior 1: logit(λΔt) ~ N(logit(p_center), scale_p²).
    Prior 2: log(σ_J/(σ√Δt)) ~ N(log(κ_center), scale_κ²).
    """
    sigma = theta[:, 1]    # (batch,) annualized diffusion vol
    lam = theta[:, 2]      # (batch,) annualized jump intensity
    sigma_j = theta[:, 4]  # (batch,) per-jump vol

    # Prior 1: p = λΔt, logit(p) ~ N(logit(p_center), scale_p²)
    p = (lam * dt).clamp(1e-6, 1 - 1e-6)
    logit_p = torch.log(p / (1 - p))
    logit_p_center = float(np.log(p_center / (1 - p_center)))
    pen_p = 0.5 * ((logit_p - logit_p_center) / scale_p) ** 2  # (batch,)

    # Prior 2: κ = σ_J / (σ√Δt), log(κ) ~ N(log(κ_center), scale_κ²)
    sigma_bar = sigma * (dt ** 0.5)
    kappa = (sigma_j / sigma_bar.clamp(min=1e-12)).clamp(min=1e-6)
    log_kappa = torch.log(kappa)
    log_kappa_center = float(np.log(kappa_center))
    pen_kappa = 0.5 * ((log_kappa - log_kappa_center) / scale_kappa) ** 2  # (batch,)

    return (pen_p + pen_kappa).mean()


def _jd_nll(
    r: torch.Tensor,
    theta_pred: torch.Tensor,
    dt: float,
    n_max: int,
) -> torch.Tensor:
    """JD NLL for theta (batch, 5)."""
    return nll(
        r, theta_pred[:, 0], theta_pred[:, 1], theta_pred[:, 2],
        theta_pred[:, 3], theta_pred[:, 4], dt, n_max,
    )


def _run_phase3_diagnostics(
    T: "TransferMap",
    theta_a_val: List[Dict[str, float]],
    theta_b_val: List[Dict[str, float]],
    r_val: np.ndarray,
    regime_seq_eth_1h: np.ndarray,
    train_end: int,
    val_end: int,
    n_regimes_eth: int,
    config: dict,
    target_asset: str = "",
) -> None:
    """
    Run diagnostics 1-3 before Phase 4. Per doc/stage6_transfer.md Section 14.
    D1: NLL(r_val, T(theta_a_val)) - Phase 3 T on val
    D2: NLL(r_val, theta_b_val) - weak model on val
    D3: param stats for T(theta_a_val), theta_b_val, theta_a_val
    """
    if not theta_a_val or not theta_b_val or len(r_val) == 0:
        print(f"  [Diagnostics] skip: no val data")
        return
    n_use = min(len(theta_a_val), len(theta_b_val), len(r_val))
    if n_use == 0:
        return

    dt = 1.0 / config["training"]["bars_per_year"]
    n_max = config["training"].get("jd_truncation_n", 10)
    lam_max = config.get("tft_jd", {}).get("lambda_max", 35040.0)
    downsample_factor = config.get("transfer", {}).get("hmm_downsample_factor", 4)

    theta_a_t = torch.FloatTensor([
        [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
         t.get("sigma_j", t.get("sigma_J", 0))]
        for t in theta_a_val[:n_use]
    ])
    theta_b_t = torch.FloatTensor([
        [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
         t.get("sigma_j", t.get("sigma_J", 0))]
        for t in theta_b_val[:n_use]
    ])
    r_t = torch.FloatTensor(r_val[:n_use])
    if not torch.isfinite(r_t).all():
        r_t = torch.nan_to_num(r_t, nan=0.0, posinf=0.0, neginf=0.0)

    regime_15min = upsample_regime_to_15min(
        regime_seq_eth_1h, val_end, downsample_factor
    )
    regime_val = regime_15min[train_end : train_end + n_use]
    regime_val_oh = F.one_hot(torch.LongTensor(regime_val), n_regimes_eth).float()

    T.eval()
    with torch.no_grad():
        theta_t_pred = T(theta_a_t, regime_val_oh)
        theta_t_pred = _clamp_theta_for_jd(theta_t_pred, lam_max)

        # D1: Phase 3 T on val
        nll_t_sum = _jd_nll(r_t, theta_t_pred, dt, n_max).item()
        nll_t_mean = nll_t_sum / n_use
        print(f"  [D1] NLL(r_val, T(theta_a_val)): mean={nll_t_mean:.4f} (n={n_use})")

        # D2: weak model theta_b_val on val
        nll_b_sum = _jd_nll(r_t, theta_b_t, dt, n_max).item()
        nll_b_mean = nll_b_sum / n_use
        print(f"  [D2] NLL(r_val, theta_b_val):   mean={nll_b_mean:.4f} (n={n_use})")

        # D3: param stats
        names = ["mu", "sigma", "lam", "mu_j", "sigma_j"]
        print(f"  [D3] Param stats (min/mean/max):")
        for i, name in enumerate(names):
            t_pred = theta_t_pred[:, i]
            t_b = theta_b_t[:, i]
            t_a = theta_a_t[:, i]
            print(f"       {name}: T(θ^a) [{t_pred.min():.4f}/{t_pred.mean():.4f}/{t_pred.max():.4f}]  "
                  f"θ^b [{t_b.min():.4f}/{t_b.mean():.4f}/{t_b.max():.4f}]  "
                  f"θ^a [{t_a.min():.4f}/{t_a.mean():.4f}/{t_a.max():.4f}]")
    T.train()


def _jd_nll_clamped(
    r: torch.Tensor,
    theta_pred: torch.Tensor,
    dt: float,
    n_max: int,
    clamp_sigma: float = 5.0,
) -> torch.Tensor:
    """Clamped JD NLL for theta (batch, 5). Prevents outlier bars from dominating gradients."""
    return nll_clamped(
        r, theta_pred[:, 0], theta_pred[:, 1], theta_pred[:, 2],
        theta_pred[:, 3], theta_pred[:, 4], dt, n_max, clamp_sigma,
    )


def _w_pt_from_theta(
    theta: torch.Tensor,
    dt: float,
    config_path: str = "config.yaml",
    return_raw: bool = False,
    gamma: Optional[float] = None,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    W_PT from stacked theta (batch, 5). Wraps compute_w_pt.
    Pass gamma/alpha explicitly to avoid repeated config file reads in training loops.
    Default return_raw=False: W_PT normalized to (0,1] by batch-max, bounded MSE and -log terms.
    Per doc/stage6_transfer.md Section 3.3 and 5.3 (compute_w_pt without return_raw=True).
    """
    return compute_w_pt(
        theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4],
        dt, gamma=gamma, alpha=alpha, cvar_method="analytic",
        config_path=config_path, return_raw=return_raw,
    )


def _jd_crps(
    r: torch.Tensor,
    theta_pred: torch.Tensor,
    dt: float,
    m_samples: int = 200,
    seed: int = 42,
) -> torch.Tensor:
    """JD CRPS via Monte Carlo."""
    return crps_mc(
        r, theta_pred[:, 0], theta_pred[:, 1], theta_pred[:, 2],
        theta_pred[:, 3], theta_pred[:, 4], dt, m_samples, seed,
    )


def finetune_transfer_map(
    T: TransferMap,
    T_phase3_state: Dict[str, torch.Tensor],
    theta_eth_15min: List[Dict[str, float]],
    theta_b_weak_train: List[Dict[str, float]],
    r_target_train: np.ndarray,
    r_target_val: np.ndarray,
    regime_seq_eth_1h: np.ndarray,
    n_regimes_eth: int,
    transfer_mode: str,
    config: dict,
    config_path: str = "config.yaml",
    theta_b_weak_val: Optional[List[Dict[str, float]]] = None,
) -> Tuple[TransferMap, int, List[Dict[str, float]]]:
    """
    Phase 4: Finetune T on target returns. Per doc/stage6_transfer.md Section 4.
    L_phase4 = L_JD + λ_wpt * L_weakness only; no L_TransWeave.
    """
    reg_config = {
        "full": {"lr": 5e-4, "epochs": 50, "patience": 10},
        "partial": {"lr": 5e-4, "epochs": 50, "patience": 10},
        "weak": {"lr": 1e-3, "epochs": 100, "patience": 15},
    }
    cfg = reg_config.get(transfer_mode, reg_config["partial"])
    tmap_cfg = config.get("transfer_map", {})

    lr = tmap_cfg.get(f"{transfer_mode}_lr", cfg["lr"])
    max_epochs = tmap_cfg.get(f"{transfer_mode}_epochs", cfg["epochs"])
    patience = tmap_cfg.get(f"{transfer_mode}_patience", cfg["patience"])
    lambda_wpt = tmap_cfg.get(f"phase4_lambda_wpt_{transfer_mode}", tmap_cfg.get("phase4_lambda_wpt", tmap_cfg.get("lambda_wpt", 0.02)))
    mu_anchor = tmap_cfg.get("phase4_l2_anchor", 50.0)
    grad_clip = tmap_cfg.get(f"phase4_grad_clip_{transfer_mode}", tmap_cfg.get("phase4_grad_clip", 1.0))
    clamp_sigma = tmap_cfg.get("phase4_clamp_sigma", config.get("tft_jd", {}).get("nll_clamp_sigma", 5.0))
    lam_max = config.get("tft_jd", {}).get("lambda_max", 35040.0)
    batch_size = tmap_cfg.get("batch_size", 2048)
    dt = 1.0 / config["training"]["bars_per_year"]
    n_max = config["training"].get("jd_truncation_n", 10)
    loss_alpha_crps = tmap_cfg.get("loss_alpha_crps", config.get("training", {}).get("loss_alpha", 1.0))
    crps_mc_samples = config.get("training", {}).get("crps_mc_samples", 200)
    downsample_factor = config.get("transfer", {}).get("hmm_downsample_factor", 4)
    # Extract W_PT params once to avoid repeated config file reads in the training loop
    risk_cfg = config.get("risk", {})
    weakness_cfg = config.get("weakness", {})
    wpt_gamma = float(risk_cfg.get("gamma", 7.0))
    wpt_alpha = float(weakness_cfg.get("cvar_alpha", risk_cfg.get("cvar_alpha", 0.05)))

    theta_a_all = torch.FloatTensor([
        [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
         t.get("sigma_j", t.get("sigma_J", 0))]
        for t in theta_eth_15min
    ])
    # θ^b_weak for L_weakness target (Phase 4 has no L_TransWeave)
    theta_b_all = torch.FloatTensor([
        [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
         t.get("sigma_j", t.get("sigma_J", 0))]
        for t in theta_b_weak_train
    ])
    # W_PT eps for normalization
    w_pt_eps = float(weakness_cfg.get("w_pt_eps", 1e-8))
    r_train_t = torch.FloatTensor(r_target_train)
    r_val_t = torch.FloatTensor(r_target_val)

    regime_15min = upsample_regime_to_15min(
        regime_seq_eth_1h, len(theta_a_all), downsample_factor
    )
    regime_oh_all = F.one_hot(torch.LongTensor(regime_15min), n_regimes_eth).float()

    n_train = min(len(theta_a_all), len(theta_b_all), len(r_train_t), len(regime_oh_all))
    theta_a_train = theta_a_all[:n_train]
    theta_b_train = theta_b_all[:n_train]
    r_train = r_train_t[:n_train]
    regime_train = regime_oh_all[:n_train]

    n_val = min(len(theta_a_all) - n_train, len(r_val_t))
    if n_val > 0:
        theta_a_val = theta_a_all[n_train : n_train + n_val]
        r_val = r_val_t[:n_val]
        regime_val = regime_oh_all[n_train : n_train + n_val]
        assert len(theta_a_val) == len(r_val) == len(regime_val), (
            f"Phase 4 data alignment: theta_a_val={len(theta_a_val)} r_val={len(r_val)} regime={len(regime_val)}"
        )
    else:
        theta_a_val = theta_a_train[-1000:]
        r_val = r_train_t[-1000:]
        regime_val = regime_train[-1000:]

    dataset = torch.utils.data.TensorDataset(theta_a_train, theta_b_train, r_train, regime_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(T.parameters(), lr=lr)

    # Snapshot Phase 3 parameters for L2 anchor (prevent drift)
    phase3_params = {name: p.clone().detach() for name, p in T.named_parameters()}

    # Helper: compute Phase 4 val loss (L_JD + λ*(-log W_PT), no L_TransWeave, no MSE consistency)
    def _phase4_val_loss(T_model: TransferMap, lam_wpt: float) -> Tuple[float, float, float, float]:
        """Returns (val_phase4_loss, val_nll, val_crps, val_weakness)."""
        T_model.eval()
        with torch.no_grad():
            theta_b_v = T_model(theta_a_val, regime_val)
            theta_b_v = _clamp_theta_for_jd(theta_b_v, lam_max)
            v_nll = _jd_nll_clamped(r_val, theta_b_v, dt, n_max, clamp_sigma).item()
            if not math.isfinite(v_nll):
                v_nll = 50.0
            v_crps = _jd_crps(r_val, theta_b_v, dt, m_samples=crps_mc_samples).item()
            v_jd = v_nll + loss_alpha_crps * v_crps

            # -log W_PT only (matching Phase 4 train loss)
            wpt_v_raw = _w_pt_from_theta(theta_b_v, dt, config_path, return_raw=True,
                                          gamma=wpt_gamma, alpha=wpt_alpha)
            wpt_v_max = wpt_v_raw.max().detach() + w_pt_eps
            wpt_v_n = (wpt_v_raw / wpt_v_max).clamp(min=1e-8, max=1.0)
            v_weak = -torch.log(wpt_v_n).mean().item()
            v_phase4 = v_jd + lam_wpt * v_weak
        return v_phase4, v_nll, v_crps, v_weak

    phase3_val_phase4, phase3_val_nll, phase3_val_crps, phase3_val_weak = _phase4_val_loss(T, lambda_wpt)
    phase3_val_JD = phase3_val_nll + loss_alpha_crps * phase3_val_crps
    T.train()
    print(f"  Phase 4 [{transfer_mode}] lr={lr} grad_clip={grad_clip} λ_wpt={lambda_wpt} μ_anchor={mu_anchor}")
    print(f"  Phase 4 [{transfer_mode}] baseline (Phase 3 T): val_phase4={phase3_val_phase4:.4f} "
          f"(val_JD={phase3_val_JD:.4f} val_weak={phase3_val_weak:.4f})")

    best_val_loss = phase3_val_phase4
    best_state = {k: v.clone() for k, v in T_phase3_state.items()}
    patience_counter = 0
    convergence_epoch = 0
    phase4_history: List[Dict[str, float]] = []

    for epoch in range(max_epochs):
        T.train()
        for batch_idx, (theta_a_batch, theta_b_batch, r_batch, regime_batch) in enumerate(loader):
            optimizer.zero_grad()
            theta_b_pred = T(theta_a_batch, regime_batch)
            theta_b_pred = _clamp_theta_for_jd(theta_b_pred, lam_max)

            # L_phase4 = L_JD + λ_wpt·L_weakness (no L_TransWeave)
            nll_loss = _jd_nll_clamped(r_batch, theta_b_pred, dt, n_max, clamp_sigma)
            crps_loss = _jd_crps(r_batch, theta_b_pred, dt, m_samples=crps_mc_samples)
            loss_jd = nll_loss + loss_alpha_crps * crps_loss

            # L_weakness = -log W_PT only (Algorithm 1 Phase 4: no MSE consistency)
            wpt_pred_raw = _w_pt_from_theta(theta_b_pred, dt, config_path, return_raw=True,
                                             gamma=wpt_gamma, alpha=wpt_alpha)
            wpt_pred_max = wpt_pred_raw.max().detach() + w_pt_eps
            wpt_pred_n = (wpt_pred_raw / wpt_pred_max).clamp(min=1e-8, max=1.0)
            loss_weakness = -torch.log(wpt_pred_n).mean()

            # L2 anchor to Phase 3 weights (prevent drift)
            l2_anchor = sum(
                (p - phase3_params[name]).pow(2).sum()
                for name, p in T.named_parameters()
            )

            # Loss decomposition logging on first batch
            if epoch == 0 and batch_idx == 0:
                s_jd_p4 = loss_jd.item()
                s_wk_p4 = loss_weakness.item()
                s_l2 = l2_anchor.item()
                print(f"  [Phase 4] epoch 0 batch 0: loss_jd={s_jd_p4:.4f} "
                      f"loss_weak(-logWPT)={s_wk_p4:.4f}(×{lambda_wpt}={lambda_wpt*s_wk_p4:.4f}) "
                      f"l2_anchor={s_l2:.6f}(×{mu_anchor}={mu_anchor*s_l2:.4f})")

            loss = loss_jd + lambda_wpt * loss_weakness + mu_anchor * l2_anchor

            loss.backward()
            grad_has_nan = any(torch.isnan(p.grad).any().item() for p in T.parameters() if p.grad is not None)
            if not grad_has_nan:
                torch.nn.utils.clip_grad_norm_(T.parameters(), grad_clip)
                optimizer.step()
            else:
                optimizer.zero_grad()

        # Val phase4 loss for early stop
        val_phase4, val_nll, val_crps, val_weak = _phase4_val_loss(T, lambda_wpt)
        val_JD = val_nll + loss_alpha_crps * val_crps
        phase4_history.append({
            "epoch": float(epoch),
            "val_phase4": float(val_phase4),
            "val_nll": float(val_nll),
            "val_crps": float(val_crps),
            "val_weak": float(val_weak),
        })

        if math.isfinite(val_phase4) and val_phase4 < best_val_loss:
            best_val_loss = val_phase4
            best_state = {k: v.clone() for k, v in T.state_dict().items()}
            patience_counter = 0
            convergence_epoch = epoch
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Phase 4 [{transfer_mode}] early stop at epoch {epoch}")
                break

        if epoch % 10 == 0 or epoch == max_epochs - 1:
            epoch_l2_anchor = sum(
                (p - phase3_params[name]).pow(2).sum().item()
                for name, p in T.named_parameters()
            )
            print(f"  Phase 4 [{transfer_mode}] epoch {epoch}: val_phase4={val_phase4:.4f} "
                  f"(val_JD={val_JD:.4f} val_weak={val_weak:.4f}) l2_anchor={epoch_l2_anchor:.6f}")

    if best_state is not None:
        T.load_state_dict(best_state)
    else:
        T.load_state_dict(T_phase3_state)
    return T, convergence_epoch, phase4_history


def compute_time_change(
    lambda_source_cumsum: np.ndarray,
    lambda_target_cumsum: np.ndarray,
) -> np.ndarray:
    """
    Formula (33): tau(t) = inf{s : int_0^s lambda^(b) du >= int_0^t lambda^(a) du}.
    Per doc/stage6_transfer.md Section 3.3, paper_desc.md 3.2.3.
    Returns int64 indices into target.
    """
    tau = np.zeros(len(lambda_source_cumsum), dtype=np.int64)
    for t in range(len(lambda_source_cumsum)):
        idx = np.searchsorted(lambda_target_cumsum, lambda_source_cumsum[t])
        tau[t] = min(idx, len(lambda_target_cumsum) - 1)
    return tau


def apply_time_change_alignment(
    theta_a: List[Dict[str, float]],
    theta_b: List[Dict[str, float]],
    r_target: np.ndarray,
    regime_15min: np.ndarray,
    needs_time_change: bool,
    dt: float,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], np.ndarray, np.ndarray]:
    """
    Align (theta^a, theta^b, r^b, regime) via time change τ when ΔΛ/T > 0.5.
    Per doc/stage6_transfer.md Section 3.3, paper_desc.md 3.2.3.
    When needs_time_change=False, returns identity (data unchanged).
    """
    if not needs_time_change:
        n = min(len(theta_a), len(theta_b), len(r_target), len(regime_15min))
        return theta_a[:n], theta_b[:n], r_target[:n], regime_15min[:n]

    lam_a = np.array([t.get("lam", t.get("lambda", 0)) for t in theta_a], dtype=np.float64) * dt
    lam_b = np.array([t.get("lam", t.get("lambda", 0)) for t in theta_b], dtype=np.float64) * dt
    lam_a = np.maximum(lam_a, 1e-12)
    lam_b = np.maximum(lam_b, 1e-12)
    lambda_source_cumsum = np.cumsum(lam_a)
    lambda_target_cumsum = np.cumsum(lam_b)
    tau = compute_time_change(lambda_source_cumsum, lambda_target_cumsum)

    theta_a_aligned = theta_a
    theta_b_aligned = [theta_b[int(tau[t])] for t in range(len(theta_a))]
    r_aligned = r_target[tau]
    regime_aligned = regime_15min[: len(theta_a)]
    return theta_a_aligned, theta_b_aligned, r_aligned, regime_aligned


def evaluate_transfer(
    T: TransferMap,
    theta_eth_15min: List[Dict[str, float]],
    r_target: np.ndarray,
    regime_seq_eth_1h: np.ndarray,
    n_regimes_eth: int,
    dt: float,
    config: dict,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Evaluate TransWeave on target returns. Per doc/stage6_transfer.md Section 7.2."""
    theta_a = torch.FloatTensor([
        [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
         t.get("sigma_j", t.get("sigma_J", 0))]
        for t in theta_eth_15min
    ])
    downsample_factor = config.get("transfer", {}).get("hmm_downsample_factor", 4)
    regime_15min = upsample_regime_to_15min(
        regime_seq_eth_1h, len(theta_a), downsample_factor
    )
    regime_oh = F.one_hot(torch.LongTensor(regime_15min), n_regimes_eth).float()
    n = min(len(theta_a), len(r_target), len(regime_oh))

    T.eval()
    n_max = config["training"].get("jd_truncation_n", 10)
    lam_max = config.get("tft_jd", {}).get("lambda_max", 35040.0)
    with torch.no_grad():
        theta_b_pred = T(theta_a[:n], regime_oh[:n])
        theta_b_pred = _clamp_theta_for_jd(theta_b_pred, lam_max)
        r_t = torch.FloatTensor(r_target[:n])
        nll_total = _jd_nll(r_t, theta_b_pred, dt, n_max).item()
        crps_val = _jd_crps(r_t, theta_b_pred, dt).item()
    return {"nll": nll_total / max(n, 1), "crps": crps_val, "n_samples": n}


def evaluate_direct(
    theta_eth_15min: List[Dict[str, float]],
    r_target: np.ndarray,
    dt: float,
    config: dict,
) -> Dict[str, Any]:
    """Direct baseline: use ETH params on target returns."""
    theta_a = torch.FloatTensor([
        [t["mu"], t["sigma"], t["lam"], t.get("mu_j", t.get("mu_J", 0)),
         t.get("sigma_j", t.get("sigma_J", 0))]
        for t in theta_eth_15min
    ])
    n = min(len(theta_a), len(r_target))
    n_max = config["training"].get("jd_truncation_n", 10)
    r_t = torch.FloatTensor(r_target[:n])
    with torch.no_grad():
        nll_total = _jd_nll(r_t, theta_a[:n], dt, n_max).item()
        crps_val = _jd_crps(r_t, theta_a[:n], dt).item()
    return {"nll": nll_total / max(n, 1), "crps": crps_val, "n_samples": n}


def evaluate_scratch(
    model: SimpleJDModel,
    asset: str,
    dt: float,
    config: dict,
    config_path: str = "config.yaml",
) -> Dict[str, Any]:
    """Scratch baseline: weak model on test split."""
    X_hist, _, y, split_arr = load_feature_arrays(asset, config_path)
    test_mask = split_arr == "test"
    X_test = X_hist[test_mask, -1, :].astype(np.float64)
    r_test_arr = y[test_mask]
    valid = np.isfinite(X_test).all(axis=1) & np.isfinite(r_test_arr)
    X_test = X_test[valid]
    r_test_arr = r_test_arr[valid]
    X_last_test = torch.FloatTensor(X_test)
    r_test = torch.FloatTensor(r_test_arr)

    model.eval()
    n_max = config["training"].get("jd_truncation_n", 10)
    with torch.no_grad():
        theta_pred = model(X_last_test)
        nll_total = _jd_nll(r_test, theta_pred, dt, n_max).item()
        crps_val = _jd_crps(r_test, theta_pred, dt).item()
    n_samples = len(r_test)
    return {"nll": nll_total / max(n_samples, 1), "crps": crps_val, "n_samples": n_samples}


def _save_phase3_checkpoint(
    ckpt_dir: Path,
    target_asset: str,
    decision: str,
    T: TransferMap,
    T_phase3_state: Dict[str, torch.Tensor],
    theta_eth_for_finetune: List[Dict[str, float]],
    theta_b_for_finetune: List[Dict[str, float]],
    r_train: np.ndarray,
    r_val: np.ndarray,
    regime_seq_eth: np.ndarray,
    n_regimes_eth: int,
    theta_eth_test: List[Dict[str, float]],
    r_test: np.ndarray,
) -> None:
    """Save Phase 3 outputs so Phase 4 can be re-run independently."""
    def _theta_list_to_array(theta_list: List[Dict[str, float]]) -> np.ndarray:
        return np.array([
            [t["mu"], t["sigma"], t["lam"],
             t.get("mu_j", t.get("mu_J", 0)), t.get("sigma_j", t.get("sigma_J", 0))]
            for t in theta_list
        ])

    save_path = ckpt_dir / f"phase3_ckpt_{target_asset}.pt"
    torch.save({
        "decision": decision,
        "T_state_dict": T.state_dict(),
        "T_phase3_state": T_phase3_state,
        "n_jd_params": T.n_jd_params,
        "n_regimes": T.n_regimes,
        "hidden_dim": T.net[0].out_features,  # hidden_dim from first layer output
        "theta_eth_finetune": _theta_list_to_array(theta_eth_for_finetune),
        "theta_b_finetune": _theta_list_to_array(theta_b_for_finetune),
        "r_train": np.asarray(r_train),
        "r_val": np.asarray(r_val),
        "regime_seq_eth": np.asarray(regime_seq_eth),
        "n_regimes_eth": n_regimes_eth,
        "theta_eth_test": _theta_list_to_array(theta_eth_test),
        "r_test": np.asarray(r_test),
    }, save_path)
    print(f"  [Phase 3 ckpt] saved to {save_path}")


def run_transfer_experiment(
    target_asset: str,
    stage5_report: dict,
    theta_eth_15min: List[Dict[str, float]],
    eth_split_indices: Dict[str, int],
    config: dict,
    config_path: str = "config.yaml",
    ckpt_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run full transfer experiment for one target. Per doc/stage6_transfer.md Section 7.3.
    """
    import json

    root = Path(__file__).resolve().parents[2]
    ckpt_dir = ckpt_dir or (root / config["paths"]["checkpoints"])
    dt = 1.0 / config["training"]["bars_per_year"]

    pair_info = next((p for p in stage5_report["pairs"] if p["target"] == target_asset), None)
    if pair_info is None:
        raise ValueError(f"No pair info for target {target_asset} in stage5 report")
    decision = pair_info["decision"]

    regime_data = np.load(ckpt_dir / "regime_ETH.npz", allow_pickle=True)
    regime_seq_eth = regime_data["state_sequence"]
    n_regimes_eth = int(regime_data["n_states"])

    r_train = load_returns(target_asset, "train", config_path)
    r_val = load_returns(target_asset, "val", config_path)
    r_test = load_returns(target_asset, "test", config_path)

    val_end = eth_split_indices["val_end"]
    theta_eth_test = theta_eth_15min[val_end:]

    results: Dict[str, Any] = {"target_asset": target_asset, "decision": decision}

    print(f"\n[{target_asset}] Training Scratch baseline...")
    scratch_model = train_weak_target_model(target_asset, config, config_path)
    results["scratch"] = evaluate_scratch(scratch_model, target_asset, dt, config, config_path)
    torch.save(scratch_model.state_dict(), ckpt_dir / f"weak_model_{target_asset}.pt")

    print(f"\n[{target_asset}] Evaluating Direct baseline...")
    results["direct"] = evaluate_direct(theta_eth_test, r_test, dt, config)

    if decision != "reject":
        print(f"\n[{target_asset}] Running TransWeave ({decision})...")
        theta_b_weak = infer_weak_theta(scratch_model, target_asset, config_path, "train")
        theta_b_val = infer_weak_theta(scratch_model, target_asset, config_path, "val")
        theta_b_test = infer_weak_theta(scratch_model, target_asset, config_path, "test")
        train_end = eth_split_indices["train_end"]
        val_end = eth_split_indices["val_end"]
        n_train = min(len(theta_eth_15min), len(theta_b_weak), train_end)
        theta_eth_train = theta_eth_15min[:n_train]
        theta_a_val = theta_eth_15min[train_end:val_end] if val_end > train_end else None
        theta_b_val_sliced = theta_b_val[: len(theta_a_val)] if (theta_a_val and theta_b_val) else None

        needs_time_change = pair_info.get("time_change", {}).get("needs_time_change", False)
        downsample_factor = config.get("transfer", {}).get("hmm_downsample_factor", 4)
        regime_15min_full = upsample_regime_to_15min(regime_seq_eth, val_end, downsample_factor)

        if needs_time_change:
            theta_eth_train, theta_b_weak_train_aligned, r_train_aligned, _ = apply_time_change_alignment(
                theta_eth_train, theta_b_weak[:n_train], r_train[:n_train],
                regime_15min_full[:n_train], True, dt,
            )
            if theta_a_val and theta_b_val_sliced is not None and len(r_val) > 0:
                n_val_use = min(len(theta_a_val), len(theta_b_val_sliced), len(r_val))
                regime_val = regime_15min_full[train_end : train_end + n_val_use]
                theta_a_val, theta_b_val_aligned, r_val_aligned, _ = apply_time_change_alignment(
                    theta_a_val[:n_val_use], theta_b_val_sliced[:n_val_use], r_val[:n_val_use],
                    regime_val, True, dt,
                )
                theta_b_val_sliced = theta_b_val_aligned
                r_val = r_val_aligned
            r_train = r_train_aligned
            theta_b_weak_train_aligned_list = theta_b_weak_train_aligned
        else:
            theta_b_weak_train_aligned_list = theta_b_weak[:n_train]

        T, phase3_history = train_transfer_map(
            theta_eth_train,
            theta_b_weak_train_aligned_list,
            r_train,
            regime_seq_eth,
            n_regimes_eth,
            config,
            config_path,
            theta_a_val=theta_a_val,
            theta_b_val=theta_b_val_sliced,
            r_target_val=r_val,
            transfer_mode=decision,
        )
        results["phase3_history"] = phase3_history
        T_phase3_state = {k: v.clone() for k, v in T.state_dict().items()}

        # Diagnostics 1-3 before Phase 4 (doc Section 14)
        _run_phase3_diagnostics(
            T, theta_a_val, theta_b_val_sliced or theta_b_val,
            r_val, regime_seq_eth, train_end, val_end, n_regimes_eth,
            config, target_asset,
        )

        theta_eth_for_finetune = theta_eth_train + (theta_a_val if theta_a_val else [])
        theta_b_for_finetune = theta_b_weak_train_aligned_list + (theta_b_val_sliced if theta_b_val_sliced else [])
        r_train_for_finetune = r_train
        r_val_for_finetune = r_val if (theta_a_val and theta_b_val_sliced) else np.array([])

        # Save Phase 3 checkpoint for Phase 4 debug iterations
        _save_phase3_checkpoint(
            ckpt_dir, target_asset, decision, T, T_phase3_state,
            theta_eth_for_finetune, theta_b_for_finetune,
            r_train_for_finetune, r_val_for_finetune,
            regime_seq_eth, n_regimes_eth,
            theta_eth_test, r_test,
        )

        T, convergence_epoch, phase4_history = finetune_transfer_map(
            T, T_phase3_state, theta_eth_for_finetune, theta_b_for_finetune,
            r_train_for_finetune, r_val_for_finetune,
            regime_seq_eth, n_regimes_eth, decision, config, config_path,
            theta_b_weak_val=theta_b_val_sliced,
        )
        results["phase4_history"] = phase4_history

        tw_result = evaluate_transfer(
            T, theta_eth_test, r_test, regime_seq_eth, n_regimes_eth, dt, config, config_path,
        )
        tw_result["convergence_epoch"] = convergence_epoch
        results["transweave"] = tw_result
        torch.save(T.state_dict(), ckpt_dir / f"transfer_map_{target_asset}.pt")
    else:
        print(f"\n[{target_asset}] Running Force-Transfer (reject validation)...")
        theta_b_weak = infer_weak_theta(scratch_model, target_asset, config_path, "train")
        theta_b_val = infer_weak_theta(scratch_model, target_asset, config_path, "val")
        theta_b_test = infer_weak_theta(scratch_model, target_asset, config_path, "test")
        train_end = eth_split_indices["train_end"]
        val_end = eth_split_indices["val_end"]
        n_train = min(len(theta_eth_15min), len(theta_b_weak), train_end)
        theta_a_val = theta_eth_15min[train_end:val_end] if val_end > train_end else None
        theta_b_val_sliced = theta_b_val[: len(theta_a_val)] if (theta_a_val and theta_b_val) else None

        needs_time_change = pair_info.get("time_change", {}).get("needs_time_change", False)
        downsample_factor = config.get("transfer", {}).get("hmm_downsample_factor", 4)
        regime_15min_full = upsample_regime_to_15min(regime_seq_eth, val_end, downsample_factor)
        if needs_time_change:
            theta_eth_train, theta_b_weak_train_aligned, r_train_aligned, _ = apply_time_change_alignment(
                theta_eth_15min[:n_train], theta_b_weak[:n_train], r_train[:n_train],
                regime_15min_full[:n_train], True, dt,
            )
            if theta_a_val and theta_b_val_sliced is not None and len(r_val) > 0:
                n_val_use = min(len(theta_a_val), len(theta_b_val_sliced), len(r_val))
                regime_val = regime_15min_full[train_end : train_end + n_val_use]
                theta_a_val, theta_b_val_aligned, r_val_aligned, _ = apply_time_change_alignment(
                    theta_a_val[:n_val_use], theta_b_val_sliced[:n_val_use], r_val[:n_val_use],
                    regime_val, True, dt,
                )
                theta_b_val_sliced = theta_b_val_aligned
                r_val = r_val_aligned
            r_train = r_train_aligned
            theta_b_weak_train_aligned_list = theta_b_weak_train_aligned
        else:
            theta_eth_train = theta_eth_15min[:n_train]
            theta_b_weak_train_aligned_list = theta_b_weak[:n_train]

        T, phase3_history = train_transfer_map(
            theta_eth_train,
            theta_b_weak_train_aligned_list,
            r_train,
            regime_seq_eth,
            n_regimes_eth,
            config,
            config_path,
            theta_a_val=theta_a_val,
            theta_b_val=theta_b_val_sliced,
            r_target_val=r_val,
            transfer_mode="weak",
        )
        results["phase3_history"] = phase3_history
        T_phase3_state = {k: v.clone() for k, v in T.state_dict().items()}
        theta_eth_for_finetune = theta_eth_train + (theta_a_val if theta_a_val else [])
        theta_b_for_finetune = theta_b_weak_train_aligned_list + (theta_b_val_sliced if theta_b_val_sliced else [])
        T, convergence_epoch, phase4_history = finetune_transfer_map(
            T, T_phase3_state, theta_eth_for_finetune, theta_b_for_finetune,
            r_train, r_val if (theta_a_val and theta_b_val_sliced) else np.array([]),
            regime_seq_eth, n_regimes_eth, "weak", config, config_path,
            theta_b_weak_val=theta_b_val_sliced,
        )
        results["phase4_history"] = phase4_history
        force_result = evaluate_transfer(
            T, theta_eth_test, r_test, regime_seq_eth, n_regimes_eth, dt, config, config_path,
        )
        force_result["convergence_epoch"] = convergence_epoch
        results["force_transfer"] = force_result
        torch.save(T.state_dict(), ckpt_dir / f"transfer_map_{target_asset}.pt")

    return results
