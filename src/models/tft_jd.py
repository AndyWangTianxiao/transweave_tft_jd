"""
TFT-JD joint model: h = TFT(X, Z) → 5 heads → (μ, σ, λ, μ_J, σ_J). Formulas (2–6).
Output heads with constraints: σ/λ/σ_J softplus+eps, μ_J Tanh×scale, λ clamp.
MAP bias initialization from eth_static_jd_params.json.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import json
import numpy as np
import torch
import torch.nn as nn
import yaml

from . import jump_diffusion as jd
from .tft import TFTEncoder


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load config from project root."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


class JDParamHeads(nn.Module):
    """
    Five independent MLP heads: h (batch, hidden_size) → (μ, σ, λ, μ_J, σ_J).
    Each head: Linear(hidden_size, 32) → ReLU → Linear(32, 1).
    Constraints per formulas (3–6): softplus+eps for σ,λ,σ_J; Tanh×scale for μ_J; λ clamp.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-4,
        mu_j_scale: float = 0.06,
        lambda_max: float = 35040,
        map_params: Optional[dict] = None,
        weight_std: float = 1e-3,
        config_path: str = "config.yaml",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.mu_j_scale = mu_j_scale
        self.lambda_max = lambda_max

        def _mlp(out_dim: int = 1) -> nn.Module:
            return nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, out_dim),
            )

        self.mu_head = _mlp()
        self.sigma_head = _mlp()
        self.lambda_head = _mlp()
        self.mu_J_head = _mlp()
        self.sigma_J_head = _mlp()

        # Init weights small, bias from MAP if provided
        for m in [self.mu_head, self.sigma_head, self.lambda_head, self.mu_J_head, self.sigma_J_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, weight_std)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        if map_params is not None:
            self._init_bias_from_map(map_params)

    def _init_bias_from_map(self, m: dict) -> None:
        """Set output head biases from MAP params. Inverse softplus for σ,λ,σ_J; arctanh for μ_J."""
        mu, sig, lam, muJ, sigJ = m["mu"], m["sigma"], m["lambda"], m["mu_J"], m["sigma_J"]
        eps = self.eps

        # mu: direct
        self._set_last_bias(self.mu_head, 0, mu)

        # sigma: inverse_softplus(sig - eps)
        inv = float(np.log(np.expm1(np.clip(sig - eps, 1e-6, None))))
        self._set_last_bias(self.sigma_head, 0, inv)

        # lambda: inverse_softplus(lam - eps)
        inv = float(np.log(np.expm1(np.clip(lam - eps, 1e-6, None))))
        self._set_last_bias(self.lambda_head, 0, inv)

        # mu_J: arctanh(muJ / mu_j_scale)
        inv = float(np.arctanh(np.clip(muJ / self.mu_j_scale, -0.999, 0.999)))
        self._set_last_bias(self.mu_J_head, 0, inv)

        # sigma_J: inverse_softplus(sigJ - eps)
        inv = float(np.log(np.expm1(np.clip(sigJ - eps, 1e-6, None))))
        self._set_last_bias(self.sigma_J_head, 0, inv)

    def _set_last_bias(self, seq: nn.Sequential, idx: int, val: float) -> None:
        """Set bias of last Linear in Sequential."""
        for layer in reversed(seq):
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                layer.bias.data[idx] = val
                return

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            h: (batch, hidden_size)
        Returns:
            mu, sigma, lam, mu_J, sigma_J: each (batch,)
        """
        mu = self.mu_head(h).squeeze(-1)
        sigma = torch.nn.functional.softplus(self.sigma_head(h).squeeze(-1)) + self.eps
        lam = (torch.nn.functional.softplus(self.lambda_head(h).squeeze(-1)) + self.eps).clamp(max=self.lambda_max)
        mu_J = torch.tanh(self.mu_J_head(h).squeeze(-1)) * self.mu_j_scale
        sigma_J = torch.nn.functional.softplus(self.sigma_J_head(h).squeeze(-1)) + self.eps
        return mu, sigma, lam, mu_J, sigma_J


class TFTJD(nn.Module):
    """
    TFT-JD joint model: TFT encoder + 5 JD param heads.
    forward(x_hist, z_future) → (μ, σ, λ, μ_J, σ_J).
    """

    def __init__(
        self,
        map_path: Optional[str] = None,
        config_path: str = "config.yaml",
    ) -> None:
        super().__init__()
        cfg = _load_config(config_path)
        tft_cfg = cfg.get("tft_jd", {})
        paths_cfg = cfg.get("paths", {})
        root = Path(__file__).resolve().parents[2]
        ckpt_dir = root / paths_cfg.get("checkpoints", "experiments/checkpoints")

        self.encoder = TFTEncoder(config_path=config_path)
        hidden_size = self.encoder.hidden_size

        eps = tft_cfg.get("softplus_eps", 1e-4)
        mu_j_scale = tft_cfg.get("mu_j_scale", 0.06)
        lambda_max = tft_cfg.get("lambda_max", 35040)
        weight_std = tft_cfg.get("head_weight_std", 1e-3)
        use_map = tft_cfg.get("head_bias_from_map", True)

        map_params = None
        if use_map and map_path is None:
            map_path = str(ckpt_dir / "eth_static_jd_params.json")
        if use_map and Path(map_path).exists():
            with open(map_path) as f:
                map_params = json.load(f)

        self.heads = JDParamHeads(
            hidden_size=hidden_size,
            eps=eps,
            mu_j_scale=mu_j_scale,
            lambda_max=lambda_max,
            map_params=map_params,
            weight_std=weight_std,
            config_path=config_path,
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        z_future: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_hist: (batch, 96, 14)
            z_future: (batch, 1, 4)
        Returns:
            mu, sigma, lam, mu_J, sigma_J: each (batch,)
        """
        h = self.encoder(x_hist, z_future)
        return self.heads(h)

    def freeze_encoder(self) -> None:
        """Freeze TFT backbone for head warmup."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze TFT backbone after warmup."""
        for p in self.encoder.parameters():
            p.requires_grad = True


def build_tft_jd(
    map_path: Optional[str] = None,
    config_path: str = "config.yaml",
) -> TFTJD:
    """Build TFT-JD model from config."""
    return TFTJD(map_path=map_path, config_path=config_path)


def infer_eth_theta(
    model: TFTJD,
    X_hist: Union[np.ndarray, torch.Tensor],
    Z_future: Union[np.ndarray, torch.Tensor],
    batch_size: int = 2048,
    device: Union[str, torch.device] = "cpu",
) -> List[dict]:
    """
    Batch inference: produce theta sequence from TFT-JD for stage6.
    Per doc/stage6_transfer.md Section 1.1.

    Args:
        model: Trained TFTJD (ETH).
        X_hist: (N, 96, 14)
        Z_future: (N, 1, 4)
        batch_size: Inference batch size.
    Returns:
        List of dicts: [{"mu", "sigma", "lam", "mu_j", "sigma_j"}, ...], annualized.
    """
    model.eval()
    model = model.to(device)
    if isinstance(X_hist, np.ndarray):
        X_hist = torch.from_numpy(X_hist.astype(np.float32))
    if isinstance(Z_future, np.ndarray):
        Z_future = torch.from_numpy(Z_future.astype(np.float32))
    X_hist = X_hist.to(device)
    Z_future = Z_future.to(device)

    n = X_hist.shape[0]
    theta_list: List[dict] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            x_b = X_hist[i:end]
            z_b = Z_future[i:end]
            mu, sigma, lam, mu_J, sigma_J = model(x_b, z_b)
            for j in range(x_b.shape[0]):
                theta_list.append({
                    "mu": mu[j].item(),
                    "sigma": sigma[j].item(),
                    "lam": lam[j].item(),
                    "mu_j": mu_J[j].item(),
                    "sigma_j": sigma_J[j].item(),
                })
    return theta_list
