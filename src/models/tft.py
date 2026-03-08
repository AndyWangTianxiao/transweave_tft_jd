"""
TFT encoder: h_t = TFT(X_t, Z_t; Ψ) ∈ ℝ^d. Formula (2).
MVP: VSN (variable selection) → 1-layer GRU → 1-layer MHA → output h.
Input: X_hist (batch, 96, 14), Z_future (batch, 1, 4).
Output: h (batch, hidden_size).
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import yaml


def _load_config(config_path: str = "config.yaml") -> dict:
    """Load config from project root."""
    root = Path(__file__).resolve().parents[2]
    with open(root / config_path) as f:
        return yaml.safe_load(f)


class VariableSelectionNetwork(nn.Module):
    """
    Lightweight variable selection: gate per input dim, then project.
    Input: (batch, seq_len, n_input)
    Output: (batch, seq_len, hidden_size)
    """

    def __init__(self, n_input: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.var_gate = nn.Linear(n_input, n_input)
        self.var_proj = nn.Linear(n_input, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_input)
        context = x.mean(dim=1)  # (batch, n_input)
        gate = torch.sigmoid(self.var_gate(context))  # (batch, n_input)
        x_gated = x * gate.unsqueeze(1)  # (batch, seq_len, n_input)
        out = self.var_proj(x_gated)  # (batch, seq_len, hidden_size)
        return self.dropout(out)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention: Q from last position, K/V from full sequence.
    For Seq2One: we predict next bar from history; use last pos as query.
    """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads = n_heads
        self.d_k = hidden_size // n_heads
        self.scale = self.d_k ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        batch, seq_len, _ = x.shape
        q = self.q_proj(x[:, -1:, :])  # (batch, 1, hidden_size) - last pos as query
        k = self.k_proj(x)  # (batch, seq_len, hidden_size)
        v = self.v_proj(x)

        # Reshape for multi-head: (batch, n_heads, seq_q, d_k)
        q = q.view(batch, 1, self.n_heads, self.d_k).transpose(1, 2)  # (batch, n_heads, 1, d_k)
        k = k.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch, n_heads, seq_len, d_k)
        v = v.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, n_heads, 1, seq_len)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (batch, n_heads, 1, d_k)
        out = out.transpose(1, 2).contiguous().view(batch, 1, -1)  # (batch, 1, hidden_size)
        out = self.out_proj(out)
        return out.squeeze(1)  # (batch, hidden_size)


class TFTEncoder(nn.Module):
    """
    TFT encoder implementing formula (2): h_t = TFT(X_t, Z_t; Ψ) ∈ ℝ^d.
    Architecture: concat Z to X → VSN → GRU → MHA → h.
    """

    def __init__(
        self,
        n_features: int = 14,
        z_dim: int = 4,
        window: int = 96,
        hidden_size: int = 64,
        attention_heads: int = 4,
        dropout: float = 0.15,
        config_path: str = "config.yaml",
    ) -> None:
        super().__init__()
        cfg = _load_config(config_path)
        tft_cfg = cfg.get("tft_jd", {})
        self.hidden_size = tft_cfg.get("hidden_size", hidden_size)
        self.attention_heads = tft_cfg.get("attention_heads", attention_heads)
        self.dropout_p = tft_cfg.get("dropout", dropout)
        stage1 = cfg.get("stage1", {})
        self.n_features = stage1.get("n_features", n_features)
        self.window = stage1.get("window", window)
        self.z_dim = z_dim

        n_input = self.n_features + self.z_dim  # 14 + 4 = 18

        self.vsn = VariableSelectionNetwork(n_input, self.hidden_size, self.dropout_p)
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.mha = MultiHeadAttention(self.hidden_size, self.attention_heads, self.dropout_p)
        self.out_dropout = nn.Dropout(self.dropout_p)

    def forward(
        self,
        x_hist: torch.Tensor,
        z_future: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_hist: (batch, window, n_features) = (batch, 96, 14)
            z_future: (batch, 1, z_dim) = (batch, 1, 4)
        Returns:
            h: (batch, hidden_size)
        """
        # Broadcast Z to all time steps and concat with X
        # z_future: (batch, 1, 4) -> (batch, window, 4)
        z_expand = z_future.expand(-1, x_hist.size(1), -1)
        x_concat = torch.cat([x_hist, z_expand], dim=-1)  # (batch, 96, 18)

        # VSN
        vsn_out = self.vsn(x_concat)  # (batch, 96, hidden_size)

        # GRU
        gru_out, _ = self.gru(vsn_out)  # (batch, 96, hidden_size)

        # MHA: Q=last, K/V=full sequence
        h = self.mha(gru_out)  # (batch, hidden_size)
        h = self.out_dropout(h)
        return h


def build_tft_encoder(config_path: str = "config.yaml") -> TFTEncoder:
    """Build TFT encoder from config."""
    cfg = _load_config(config_path)
    tft_cfg = cfg.get("tft_jd", {})
    stage1 = cfg.get("stage1", {})
    return TFTEncoder(
        n_features=stage1.get("n_features", 14),
        z_dim=4,
        window=stage1.get("window", 96),
        hidden_size=tft_cfg.get("hidden_size", 64),
        attention_heads=tft_cfg.get("attention_heads", 4),
        dropout=tft_cfg.get("dropout", 0.15),
        config_path=config_path,
    )
