#!/usr/bin/env python3
"""
Build the reproduction report notebook (reproduction_report.ipynb).
Run: python report/build_report_notebook.py
"""
import json
from pathlib import Path

REPORT_DIR = Path(__file__).parent
ROOT = REPORT_DIR.parent


def md(source: str) -> dict:
    """Markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().split("\n")}


def code(source: str, hidden: bool = True) -> dict:
    """Code cell. hidden=True means tagged for --no-input export."""
    meta = {"tags": ["remove_input"]} if hidden else {}
    lines = source.strip().split("\n")
    # Add newlines except last
    lines = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": meta, "source": lines, "execution_count": None, "outputs": []}


cells = []

# ============================================================
# Title & Abstract
# ============================================================
cells.append(md("""
# Reproducing TransWeave: A Unified Framework for Jump-Diffusion Modeling and Transfer Learning in Financial Markets

**Original paper**: Goertzel (2025), *A Unified Framework for Jump-Diffusion Modeling and Transfer Learning in Financial Markets with Behavioral Risk Measures*

**Source asset**: ETH &nbsp;|&nbsp; **Target assets**: BTC, SOL, DOGE
**Data**: Hourly OHLCV + on-chain features, Jan 2022 – Dec 2025
**Framework**: PyTorch 2.x, TFT encoder, Merton JD density

---
"""))

# ============================================================
# Setup cell
# ============================================================
cells.append(code("""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    plt.style.use("seaborn-v0_8-paper")
except Exception:
    pass
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

ROOT = Path(".").resolve().parent  # code/
CKPT = ROOT / "experiments" / "checkpoints"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "GBM": "#999999", "Static-JD": "#4DBEEE", "TFT-JD": "#0072BD", "TFT-JD+W": "#D95319",
    "Scratch": "#EDB120", "Direct": "#7E2F8E", "TransWeave": "#77AC30", "Target-TFT": "#A2142F",
}
FIGSIZE = (10, 4.5)
DPI = 150

def load_json(name):
    with open(CKPT / name) as f:
        return json.load(f)

print("Setup OK. ROOT =", ROOT)
"""))

# ============================================================
# LAYER 1: Core Results
# ============================================================
cells.append(md("""
---

# 1. Core Results

## 1.1 ETH Source Model Ablation

We train four increasingly sophisticated models on ETH hourly returns and evaluate on a held-out test set (8,736 bars). Metrics: negative log-likelihood (NLL, lower is better), CRPS, and VaR 5% breach rate (target: 5%).
"""))

cells.append(code("""
# --- Table 1: ETH Ablation ---
abl = load_json("stage8a_eth_ablation.json")
models = list(abl["models"].keys())
rows = []
for m in models:
    d = abl["models"][m]
    rows.append({
        "Model": m,
        "NLL": f'{d["nll"]:.3f}',
        "CRPS": f'{d["crps"]:.5f}',
        "VaR 5% Breach": f'{d.get("var_breach_rate", float("nan"))*100:.1f}%' if "var_breach_rate" in d else "—",
        "PIT KS p-value": f'{d.get("pit_ks_pvalue", float("nan")):.4f}' if "pit_ks_pvalue" in d else "—",
    })
df_abl = pd.DataFrame(rows)
print(df_abl.to_markdown(index=False))
"""))

cells.append(md("""
**Table 1.** ETH source-model ablation on test set. TFT-JD achieves the best NLL (−4.343), improving over Static-JD by 0.19 nats. TFT-JD+W (with weakness regularization) trades a small NLL penalty (−4.312) for behavioral calibration. All JD models dramatically outperform the GBM baseline. The VaR breach rate for TFT-JD (5.6%) is closest to the nominal 5%.
"""))

# --- Fig 1: Ablation bar chart ---
cells.append(code("""
# --- Figure 1: ETH Ablation NLL & CRPS ---
abl = load_json("stage8a_eth_ablation.json")
models = list(abl["models"].keys())
nll_vals = [abl["models"][m]["nll"] for m in models]
crps_vals = [abl["models"][m]["crps"] for m in models]
colors = [COLORS.get(m, "#666") for m in models]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
x = np.arange(len(models))
ax1.bar(x, nll_vals, color=colors, edgecolor="white", linewidth=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=9)
ax1.set_ylabel("NLL (lower is better)"); ax1.set_title("(a) Negative Log-Likelihood")
for i, v in enumerate(nll_vals):
    ax1.text(i, v - 0.02, f"{v:.3f}", ha="center", va="top", fontsize=8)

ax2.bar(x, crps_vals, color=colors, edgecolor="white", linewidth=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=9)
ax2.set_ylabel("CRPS (lower is better)"); ax2.set_title("(b) CRPS")
for i, v in enumerate(crps_vals):
    ax2.text(i, v + 0.00002, f"{v:.5f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Figure 1: ETH Source Model Ablation", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "report_fig1_ablation.png", dpi=DPI, bbox_inches="tight")
plt.show()
"""))

# ============================================================
# 1.2 Transfer Results
# ============================================================
cells.append(md("""
## 1.2 Transfer Learning Results

We apply TransWeave to transfer the ETH source model to three target assets. The transfer decision (Full / Partial / Weak) is determined by the regime correlation $\\rho$ and composite score $S_{\\text{transfer}}$ (Eq. 12–13, 58). We compare four approaches:

- **Scratch**: Target-specific JD with random initialization
- **Direct**: Directly applying the ETH model without adaptation
- **TransWeave**: Transfer via learned parameter map $T$ with weakness regularization
- **Target-TFT**: Fully trained TFT-JD on the target asset (upper bound)
"""))

cells.append(code("""
# --- Table 2: Transfer Results ---
s5 = load_json("stage5_transfer_report.json")
s6 = load_json("stage6_experiment_results.json")
tw_vs = load_json("stage8a_transweave_vs_target_tft.json")
tgt_tft = load_json("stage8a_target_tft.json")

s5_by = {p["target"]: p for p in s5["pairs"]}
tw_by = tw_vs["by_target"]

rows = []
for r in s6:
    t = r["target_asset"]
    p = s5_by.get(t, {})
    tt = tw_by.get(t, {})
    tft_nll = tgt_tft["target_tft_jd"].get(t, {}).get("nll", float("nan"))
    rows.append({
        "Target": t,
        "Decision": r["decision"].capitalize(),
        "ρ": f'{p.get("rho", 0):.3f}',
        "S_transfer": f'{p.get("s_transfer", 0):.3f}',
        "Scratch NLL": f'{r["scratch"]["nll"]:.3f}',
        "TransWeave NLL": f'{r["transweave"]["nll"]:.3f}',
        "Target-TFT NLL": f'{tft_nll:.3f}',
        "Δ(TW−TFT)": f'{tt.get("delta_nll", 0):+.3f}',
        "Grade": tt.get("grade", "?"),
    })
df_transfer = pd.DataFrame(rows)
print(df_transfer.to_markdown(index=False))
"""))

cells.append(md("""
**Table 2.** Transfer learning results. TransWeave improves over Scratch for BTC (−0.072 NLL) and DOGE (−0.046 NLL). For SOL, TransWeave slightly underperforms Scratch (+0.035), consistent with its lower $S_{\\text{transfer}}$ (0.391). Compared to the fully-trained Target-TFT upper bound, TransWeave achieves "good" (BTC), "acceptable" (SOL), and "excellent" (DOGE) grades.
"""))

# --- Fig 2: Transfer NLL grouped bar ---
cells.append(code("""
# --- Figure 2: Transfer NLL Comparison ---
s6 = load_json("stage6_experiment_results.json")
tgt_tft = load_json("stage8a_target_tft.json")
targets = [r["target_asset"] for r in s6]

methods = ["Scratch", "Direct", "TransWeave", "Target-TFT"]
nll_data = {m: [] for m in methods}
for r in s6:
    t = r["target_asset"]
    nll_data["Scratch"].append(r["scratch"]["nll"])
    nll_data["Direct"].append(r["direct"]["nll"])
    nll_data["TransWeave"].append(r["transweave"]["nll"])
    nll_data["Target-TFT"].append(tgt_tft["target_tft_jd"][t]["nll"])

fig, ax = plt.subplots(figsize=FIGSIZE)
x = np.arange(len(targets))
n = len(methods)
w = 0.8 / n
for i, m in enumerate(methods):
    bars = ax.bar(x + i * w, nll_data[m], w, label=m, color=COLORS.get(m, "#666"), edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, nll_data[m]):
        ax.text(bar.get_x() + bar.get_width()/2, val - 0.02, f"{val:.2f}", ha="center", va="top", fontsize=7, rotation=90)

ax.set_xticks(x + w * (n-1) / 2)
ax.set_xticklabels(targets)
ax.set_ylabel("NLL (lower is better)")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Figure 2: Transfer NLL — Scratch vs Direct vs TransWeave vs Target-TFT")
plt.tight_layout()
plt.savefig(FIG_DIR / "report_fig2_transfer_nll.png", dpi=DPI, bbox_inches="tight")
plt.show()
"""))

# --- Fig 3: S_transfer vs ΔNLL scatter ---
cells.append(code("""
# --- Figure 3: S_transfer vs NLL improvement ---
corr = load_json("stage8a_s_correlation.json")
fig, ax = plt.subplots(figsize=(7, 5))
for p in corr["pairs"]:
    ax.scatter(p["s_transfer"], p["delta_nll"], s=120, color=COLORS["TransWeave"], zorder=5)
    ax.annotate(p["target"], (p["s_transfer"], p["delta_nll"]), xytext=(8, 8),
                textcoords="offset points", fontsize=10, fontweight="bold")
ax.axhline(0, color="gray", ls="--", alpha=0.5)

# Add correlation annotation
ax.text(0.05, 0.95, f'Spearman r = {corr["spearman_r"]:.2f}\\nPearson r = {corr["pearson_r"]:.2f}',
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

ax.set_xlabel("$S_{\\\\mathrm{transfer}}$ (higher → more transferable)")
ax.set_ylabel("$\\\\Delta$ NLL (TransWeave − Scratch, negative is better)")
ax.set_title("Figure 3: Transfer Score vs NLL Improvement")
plt.tight_layout()
plt.savefig(FIG_DIR / "report_fig3_s_vs_nll.png", dpi=DPI, bbox_inches="tight")
plt.show()
"""))

# ============================================================
# 1.3 Key Findings
# ============================================================
cells.append(md("""
## 1.3 Key Findings

1. **Jump-Diffusion substantially outperforms GBM**: The Static-JD model improves NLL by 0.04 nats over GBM; conditioning on features via TFT yields a further 0.19 nats improvement, confirming that time-varying JD parameters capture crypto market dynamics effectively.

2. **Transfer efficacy correlates with $S_{\\text{transfer}}$**: The composite transfer score (Spearman $r = -1.0$ with $\\Delta$NLL) correctly ranks asset transferability. BTC ($S = 0.57$) benefits most; SOL ($S = 0.39$) least — consistent with SOL's higher structural divergence from ETH.

3. **Weakness regularization as behavioral Occam's razor**: The prospect-theoretic weakness measure $W_{PT}$ (Eq. 36–37) provides meaningful regularization. While TFT-JD+W trades ~0.03 NLL for behavioral calibration on the source, its transfer maps produce competitive or superior results on targets.

4. **$\\lambda$–$\\sigma_J$ identifiability**: MLE tends toward a high-$\\lambda$, low-$\\sigma_J$ ridge. MAP priors (Stage 8b ablation) break this degeneracy, yielding more interpretable jump parameters without sacrificing fit quality.

5. **Implementation insights**: (i) Using batch-mean (not sum) for $L_{\\text{unified}}$ was critical for stable training; (ii) $\\text{Ent}(T)$ regularization enforces isometry, not variance maximization; (iii) Phase 4 fine-tuning required careful $\\lambda_{wpt}$ scheduling (0.5→0.1) to avoid sharp minima from Phase 3.
"""))

# ============================================================
# LAYER 2: Implementation Transparency
# ============================================================
cells.append(md("""
---

# 2. Implementation Details

## 2.1 Jump-Diffusion Model and Density (Stage 2)

The core model follows Merton's jump-diffusion (Eq. 1):

$$r_{t+1} = \\mu_t \\Delta t + \\sigma_t \\sqrt{\\Delta t} \\, \\varepsilon_t + \\sum_{k=1}^{N_t} Y_{t,k}$$

where $N_t \\sim \\text{Poisson}(\\lambda_t \\Delta t)$ and $Y_{t,k} \\sim \\mathcal{N}(\\mu_J, \\sigma_J^2)$.

**Log-density** (Eq. 8): We truncate the Poisson sum at $n_{\\max} = 10$ terms:

$$\\log p(r | \\theta) = \\log \\sum_{n=0}^{n_{\\max}} \\frac{(\\lambda \\Delta t)^n e^{-\\lambda \\Delta t}}{n!} \\cdot \\phi\\bigl(r; \\, \\mu \\Delta t + n \\mu_J, \\, \\sigma^2 \\Delta t + n \\sigma_J^2\\bigr)$$

**Static MAP estimation** (Scheme 2): We add a Beta prior on $p_{\\text{jump}} = 1 - e^{-\\lambda \\Delta t}$ centered at empirically-estimated jump frequency (BNS + Lee-Mykland estimators), breaking the $\\lambda$–$\\sigma_J$ ridge.
"""))

cells.append(code("""
# --- Table 3: Static JD Parameters ---
assets = ["ETH", "BTC", "SOL", "DOGE"]
rows = []
for a in assets:
    try:
        p = load_json(f"{a}_static_jd_params.json")
    except FileNotFoundError:
        p = load_json(f"{a.lower()}_static_jd_params.json")
    rows.append({"Asset": a, "μ": f'{p["mu"]:.3f}', "σ": f'{p["sigma"]:.3f}',
                 "λ": f'{p["lambda"]:.1f}', "μ_J": f'{p["mu_J"]:.5f}', "σ_J": f'{p["sigma_J"]:.4f}'})
df_jd = pd.DataFrame(rows)
print("**Table 3.** Static JD MAP parameters (annualized).")
print(df_jd.to_markdown(index=False))
"""))

# ============================================================
# 2.2 TFT-JD Conditioning
# ============================================================
cells.append(md("""
## 2.2 TFT-JD Neural Conditioning (Stage 3)

The TFT encoder maps historical features $X_t$ (window=96 bars) and known future covariates $Z_t$ (hour/day-of-week sinusoids) to a hidden representation $h_t$. Prediction heads produce time-varying JD parameters:

- $\\mu_t = f_\\mu(h_t)$ — linear head
- $\\sigma_t = \\text{softplus}(f_\\sigma(h_t))$ — ensures positivity (Eq. 4)
- $\\lambda_t = \\text{softplus}(f_\\lambda(h_t))$ — ensures positivity (Eq. 5)
- $\\mu_J, \\sigma_J$ — additional heads (Eq. 6)

**Training loss** (Eq. 7): JD NLL + optional $\\lambda$-prior regularization.

**Three training modes**: base (NLL only), scratch+W (NLL + weakness from random init), finetune+W (NLL + weakness from base checkpoint).

**Key implementation choices**:
- `bars_per_year = 8766` (365.25 × 24), so $\\Delta t \\approx 1.14 \\times 10^{-4}$ years
- Softplus ensures $\\sigma_t, \\lambda_t > 0$ without gradient-killing clamps
- $\\lambda$-prior weight = 0.05 in TFT training (breaks MLE ridge)
"""))

# ============================================================
# 2.3 Weakness Regularization
# ============================================================
cells.append(md("""
## 2.3 Prospect-Theoretic Weakness (Stage 4)

**Prelec weighting** (Eq. 35): $\\pi(p) = \\exp(-(-\\ln p)^\\delta)$ with $\\delta = 0.65$ (Tversky-Kahneman calibration).

**Value function**: $v(m) = \\text{softplus-normalized gain/loss}$ with $\\alpha = 0.88$, $\\beta = 0.88$, $\\lambda_{\\text{loss}} = 2.25$.

**Weakness measure** (Eq. 36–37):

$$W_{PT}(\\theta) = \\pi(p_{\\text{jump}}) \\cdot v(m_{\\text{jump}})$$

where $p_{\\text{jump}} = 1 - e^{-\\lambda \\Delta t}$ and $m_{\\text{jump}} = |\\mu_J| + \\sigma_J$.

**Phase 3 loss** (Eq. 50): $L_{\\text{weakness}} = -\\log W_{PT} + \\text{MSE}(\\hat{W}_{PT}, W_{PT}^{\\text{target}})$ — joint A1 normalization.

**Phase 4 loss**: Only $-\\log W_{PT}$ term (Algorithm 1 gradient: $\\nabla L = \\nabla L_{\\text{data}} - \\beta \\nabla \\log W_{PT}$).
"""))

# ============================================================
# 2.4 Regime & Transfer Metrics
# ============================================================
cells.append(md("""
## 2.4 Regime Identification and Transfer Metrics (Stage 5)

**HMM regime detection**: 2-state Gaussian HMM fitted on train-set PCA features. Produces regime sequences for each asset.

**Wasserstein JD distance** $W_{JD}$ (Eq. 12): Rolling-window comparison of source vs target JD parameter distributions.

**Regime correlation** $\\rho_{\\text{regime}}$ (Eq. 13): Spectral coherence between source and target regime sequences.

**Critical threshold** $W_{\\text{crit}} = 2\\sqrt{\\bar{\\sigma}^2 + \\bar{\\lambda}}$: Transfer feasibility boundary.

**Composite score** $S_{\\text{transfer}}$ (Eq. 58): Combines $W_{JD}/W_{\\text{crit}}$, $\\rho$, and $\\Delta W_{PT}$ into a single transferability metric.

**Transfer decision**:
| Condition | Decision |
|-----------|----------|
| $\\rho > \\rho_{\\text{full}}$ (0.75) and $\\Delta W_{PT} < 0.05$ | Full |
| $\\rho > \\rho_{\\text{partial}}$ (0.6) | Partial |
| Otherwise | Weak |
"""))

cells.append(code("""
# --- Table 4: Transfer Metrics ---
s5 = load_json("stage5_transfer_report.json")
rows = []
for p in s5["pairs"]:
    rows.append({
        "Pair": f'ETH→{p["target"]}',
        "W_JD_eff": f'{p["w_jd_effective"]:.2f}',
        "W_crit": f'{p["w_crit"]:.1f}',
        "W_JD/W_crit": f'{p["w_jd_over_w_crit"]:.3f}',
        "ρ": f'{p["rho"]:.3f}',
        "ΔW_PT (p90)": f'{p["delta_wpt_p90"]:.4f}',
        "S_transfer": f'{p["s_transfer"]:.3f}',
        "Decision": p["decision"].capitalize(),
    })
df_s5 = pd.DataFrame(rows)
print("**Table 4.** Stage 5 transfer feasibility metrics.")
print(df_s5.to_markdown(index=False))
"""))

# ============================================================
# 2.5 TransWeave Transfer (Stage 6)
# ============================================================
cells.append(md("""
## 2.5 TransWeave Transfer Learning (Stage 6)

**Transfer map** $T: \\Theta^{(a)} \\to \\Theta^{(b)}$: A 3-layer MLP (128→64→5) mapping source JD parameters to target space.

**Unified loss** (Algorithm 1):

$$L_{\\text{unified}} = L_{JD}(r^{(b)} | T(\\theta^{(a)})) + \\lambda_{TW} \\cdot L_{TW}(T) + \\lambda_{wpt} \\cdot (-\\log W_{PT})$$

where $L_{TW} = \\text{Ent}(T) + \\text{smoothness}$ enforces isometry on $T$.

**Two-phase training**:
- **Phase 3**: Joint optimization of $T$ + weak target model. $L = L_{JD} + \\lambda_{TW} L_{TW} + \\lambda_{wpt}(-\\log W_{PT}) + \\mu \\|T - T_0\\|^2$
- **Phase 4**: Fine-tuning with anchored $L_2$ regularization: $L = L_{JD} + \\lambda_{wpt}(-\\log W_{PT}) + \\mu \\|T - T_{p3}\\|^2$

**Mode-dependent** $\\lambda_{TW}$: Full=0.1, Partial=0.05, Weak=0.025.

**Key implementation details**:
- **Batch-mean normalization**: Using per-sample mean (not sum) for $L_{\\text{unified}}$ was critical — sum caused gradient explosion
- **Ent(T) = isometry**: We penalize deviation from unit singular values, not entropy maximization
- **B2 parameter normalization**: `set_norm_stats()` standardizes each JD parameter dimension independently
- **Time change**: Condition (29) checked; all pairs satisfy $|\\Delta\\lambda_{\\text{ratio}}| < 0.01$, so no time-change correction needed
"""))

# ============================================================
# 2.6 Feature Engineering
# ============================================================
cells.append(md("""
## 2.6 Data and Feature Engineering

**Assets**: ETH (source), BTC, SOL, DOGE (targets). Hourly OHLCV from CCXT (Binance).

**Features** (20 dimensions): log returns, realized volatility (1h/6h/24h), RSI(14), volume-weighted return, return skewness, volume ratio, plus on-chain metrics where available (active addresses, transaction count, exchange flows).

**Time covariates** $Z_t$: hour-of-day and day-of-week sinusoid encodings (4 dims).

**Train/Val/Test split**: 2022-01-01 to 2024-06-30 / 2024-07-01 to 2024-12-31 / 2025-01-01 to 2025-12-31.

**No data leakage**: All rolling features use causal windows; standardization fitted on train only; HMM fitted on train only. Verified by independent audit (`leakage_audit.json`: all 5 checks PASS).
"""))

# ============================================================
# LAYER 3: Evaluation & Diagnostics
# ============================================================
cells.append(md("""
---

# 3. Evaluation and Diagnostics

## 3.1 Phase 3/4 Training Curves
"""))

cells.append(code("""
# --- Figure 4: Phase 3/4 training curves ---
s6 = load_json("stage6_experiment_results.json")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for idx, r in enumerate(s6):
    ax = axes[idx]
    t = r["target_asset"]
    # Phase 3
    p3 = r.get("phase3_history", [])
    if p3:
        epochs = [h["epoch"] for h in p3]
        val_jd = [h["val_JD"] for h in p3]
        ax.plot(epochs, val_jd, "b-", label="Phase 3 val_JD", linewidth=1.5)

    # Phase 4
    p4 = r.get("phase4_history", [])
    if p4:
        offset = epochs[-1] + 1 if epochs else 0
        p4_epochs = [offset + h["epoch"] for h in p4]
        p4_val_jd = [h.get("val_nll", h.get("val_JD", h.get("val_phase4", 0))) for h in p4]
        ax.plot(p4_epochs, p4_val_jd, "r-", label="Phase 4 val_JD", linewidth=1.5)
        ax.axvline(offset, color="gray", ls=":", alpha=0.5, label="Phase 3→4")

    ax.set_title(f"{t} ({r['decision']})", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val NLL")
    ax.legend(fontsize=7, loc="upper right")

fig.suptitle("Figure 4: Phase 3/4 Training Curves (Val JD NLL)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "report_fig4_training_curves.png", dpi=DPI, bbox_inches="tight")
plt.show()
"""))

# ============================================================
# 3.2 VaR Backtest
# ============================================================
cells.append(md("""
## 3.2 VaR Backtest
"""))

cells.append(code("""
# --- Figure 5: VaR Breach Rate ---
abl = load_json("stage8a_eth_ablation.json")
models_var = [m for m in abl["models"] if "var_breach_rate" in abl["models"][m]]
rates = [abl["models"][m]["var_breach_rate"] * 100 for m in models_var]
colors = [COLORS.get(m, "#666") for m in models_var]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(models_var))
ax.bar(x, rates, color=colors, edgecolor="white", linewidth=0.5)
ax.axhline(5.0, color="red", ls="--", alpha=0.7, label="Expected 5%")
ax.set_xticks(x); ax.set_xticklabels(models_var, fontsize=10)
ax.set_ylabel("VaR 5% Breach Rate (%)")
for i, v in enumerate(rates):
    ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
ax.legend()
ax.set_title("Figure 5: VaR 5% Backtest (ETH Test Set)")
plt.tight_layout()
plt.savefig(FIG_DIR / "report_fig5_var_backtest.png", dpi=DPI, bbox_inches="tight")
plt.show()
"""))

# ============================================================
# 3.3 Theorem 5.1 Verification
# ============================================================
cells.append(md("""
## 3.3 Theorem 5.1 Verification

Theorem 5.1 states that if (1) statistical compatibility ($W_{JD} < W_{\\text{crit}}$), (2) structural alignment ($\\rho > \\tau_{\\text{crit}}$), and (3) behavioral consistency ($\\Delta W_{PT}$ bounded) all hold, then TransWeave should improve over scratch. We verify:
"""))

cells.append(code("""
# --- Table 5: Theorem 5.1 ---
thm = load_json("stage8a_theorem_verification.json")
rows = []
for p in thm["pairs"]:
    rows.append({
        "Target": p["target"],
        "Cond 1 (W_JD)": "✓" if p["condition_1_w_jd"] else "✗",
        "Cond 2 (ρ)": "✓" if p["condition_2_rho"] else "✗",
        "Cond 3 (spectral)": "✓" if p["condition_3_spectral"] else "✗",
        "All Met": "✓" if p["all_conditions"] else "✗",
        "TW > Scratch?": "✓" if p["transweave_success"] else "✗",
        "Scratch NLL": f'{p["scratch_nll"]:.3f}',
        "TW NLL": f'{p["transweave_nll"]:.3f}',
    })
df_thm = pd.DataFrame(rows)
print("**Table 5.** Theorem 5.1 verification.")
print(df_thm.to_markdown(index=False))
"""))

cells.append(md("""
All three pairs satisfy all conditions. TransWeave succeeds for BTC and DOGE (2/3 = 67%). SOL is the exception: despite satisfying all conditions, TransWeave slightly underperforms Scratch (+0.035 NLL), suggesting the theorem's sufficient conditions are not always tight in practice — consistent with SOL's lowest $S_{\\text{transfer}}$ and highest $W_{JD}$.
"""))

# ============================================================
# 3.4 σ_t and λ_t dynamics (requires model inference)
# ============================================================
cells.append(md("""
## 3.4 Parameter Dynamics ($\\sigma_t$, $\\lambda_t$)
"""))

cells.append(code("""
# --- Figure 6: σ_t and λ_t dynamics from ETH TFT-JD ---
import yaml
sys.path.insert(0, str(ROOT))
try:
    import torch
    from src.models.tft_jd import TFTJD, infer_eth_theta

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    ckpt_path = CKPT / "eth_tft_jd.ckpt"
    feat_path = ROOT / "data" / "features" / "ETH_tft_arrays.npz"

    if ckpt_path.exists() and feat_path.exists():
        model = TFTJD(config_path=str(ROOT / "config.yaml"))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        data = np.load(feat_path)
        X_hist = data["X_hist"]
        Z_future = data["Z_future"]
        split = data["split"]
        test_mask = split == 2  # test set

        theta_list = infer_eth_theta(model, X_hist[test_mask][:2000], Z_future[test_mask][:2000])
        sigma_t = np.array([t["sigma"] for t in theta_list])
        lambda_t = np.array([t["lam"] for t in theta_list])
        t_axis = np.arange(len(sigma_t))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax1.plot(t_axis, sigma_t, color=COLORS["TFT-JD"], linewidth=0.6)
        ax1.set_ylabel("σ_t (annualized)")
        ax1.set_title("(a) Diffusion Volatility σ_t")
        ax1.axhline(sigma_t.mean(), color="gray", ls="--", alpha=0.5, label=f"mean={sigma_t.mean():.3f}")
        ax1.legend(fontsize=8)

        ax2.plot(t_axis, lambda_t, color=COLORS["TFT-JD+W"], linewidth=0.6)
        ax2.set_ylabel("λ_t (annualized)")
        ax2.set_xlabel("Test set bar index (hourly)")
        ax2.set_title("(b) Jump Intensity λ_t")
        ax2.axhline(lambda_t.mean(), color="gray", ls="--", alpha=0.5, label=f"mean={lambda_t.mean():.1f}")
        ax2.legend(fontsize=8)

        fig.suptitle("Figure 6: ETH Time-Varying JD Parameters (TFT-JD, Test Set)", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "report_fig6_param_dynamics.png", dpi=DPI, bbox_inches="tight")
        plt.show()
    else:
        print("[INFO] Checkpoint or features not found; skipping σ_t/λ_t plot.")
except Exception as e:
    print(f"[INFO] Could not generate parameter dynamics: {e}")
"""))

# ============================================================
# 3.5 Leakage Audit
# ============================================================
cells.append(md("""
## 3.5 Data Leakage Audit

An independent audit checked five potential leakage vectors:
"""))

cells.append(code("""
# --- Table 6: Leakage Audit ---
audit = load_json("leakage_audit.json")
rows = []
for item in audit["items"]:
    rows.append({
        "Check": item["item"],
        "Location": item["location"],
        "Result": item["result"],
        "Details": item["details"][:80] + "..." if len(item["details"]) > 80 else item["details"],
    })
df_audit = pd.DataFrame(rows)
print(f'**Table 6.** Leakage audit — {audit["critical_issues"]} critical issues found.')
print(df_audit.to_markdown(index=False))
"""))

# ============================================================
# 3.6 Lambda Ablation (Stage 8b)
# ============================================================
cells.append(md("""
## 3.6 $\\lambda$ Prior Ablation (Stage 8b)

We compare MAP estimation (with empirical $\\lambda$ prior) against pure MLE, both for static JD and TFT-JD. The MAP prior breaks the $\\lambda$–$\\sigma_J$ identifiability ridge, producing more interpretable parameters without sacrificing fit quality.
"""))

cells.append(code("""
# --- Table 7: Lambda Ablation ---
lab = load_json("stage8b_lambda_ablation.json")
rows = []
# Static
for key, label in [("A_map", "Static-JD (MAP)"), ("B_mle", "Static-JD (MLE)")]:
    pf = lab["static"][key]["params_file"]
    try:
        p = load_json(pf)
        rows.append({"Model": label, "λ": f'{p["lambda"]:.1f}', "σ_J": f'{p["sigma_J"]:.4f}',
                     "μ_J": f'{p["mu_J"]:.5f}', "Prior Weight": str(lab["static"][key].get("prior_weight", "—"))})
    except Exception:
        rows.append({"Model": label, "λ": "—", "σ_J": "—", "μ_J": "—", "Prior Weight": "—"})
# TFT
for key, label in [("A_prior", "TFT-JD (λ-prior)"), ("B_no_prior", "TFT-JD (no prior)")]:
    rows.append({"Model": label, "λ": "dynamic", "σ_J": "dynamic",
                 "μ_J": "dynamic", "Prior Weight": str(lab["tft"][key].get("lambda_prior_weight", "—"))})
df_lab = pd.DataFrame(rows)
print("**Table 7.** λ prior ablation.")
print(df_lab.to_markdown(index=False))
"""))

# ============================================================
# 4. Limitations and Future Work
# ============================================================
cells.append(md("""
---

# 4. Limitations and Future Work

1. **Hyperparameter sensitivity**: Key hyperparameters ($\\lambda_{TW}$, $\\lambda_{wpt}$, Phase 3/4 learning rates, patience) were tuned manually. A systematic search (e.g., Bayesian optimization) could improve results, especially for SOL.

2. **$\\lambda$–$\\sigma_J$ identifiability**: While MAP priors mitigate the ridge, the fundamental non-identifiability persists. Alternative jump-size distributions (e.g., double-exponential) may help.

3. **Transfer map does not use target features $X^{(b)}$**: The current $T(\\theta^{(a)})$ operates purely in parameter space. Incorporating target-side features could improve adaptation, at the cost of additional complexity.

4. **Single source asset**: We only test ETH→{BTC, SOL, DOGE}. Multi-source transfer and reverse-direction experiments remain unexplored.

5. **Architecture comparison**: We use TFT exclusively. Comparing with LSTM, Transformer, or simple MLP encoders would clarify how much of the improvement comes from the JD formulation vs. the encoder architecture.

6. **Sample period**: Results are based on 2022–2025 crypto data, which includes both bear and bull regimes but may not generalize to other asset classes or time periods.
"""))

# ============================================================
# 5. Reproducibility
# ============================================================
cells.append(md("""
---

# 5. Reproducibility

**Environment**: Python 3.10+, PyTorch 2.x, conda environment `transweave`.

**Quick evaluation** (when checkpoints exist):
```bash
python scripts/run_unified.py --mode skip
python scripts/run_stage8a_eval.py --skip-target-train
```

**Full retraining** (~9 hours on Apple M-series):
```bash
bash scripts/run_full_pipeline.sh
```

**Configuration**: All hyperparameters in `config.yaml`. No hardcoded values in source code.

**Code structure**:
| Module | Paper Section |
|--------|--------------|
| `src/models/jump_diffusion.py` | §2 (Eq. 1, 8) |
| `src/models/tft_jd.py` | §2 (Eq. 2–6) |
| `src/behavioral/weakness.py` | §4 (Eq. 36–37) |
| `src/transfer/transweave.py` | §3 (Eq. 9–11, 49), Algorithm 1 |
| `src/transfer/metrics.py` | §3 (Eq. 12–13, 58) |
| `src/unified/framework.py` | Algorithm 1 |
"""))

# ============================================================
# Build notebook JSON
# ============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (transweave)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

out_path = REPORT_DIR / "reproduction_report.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Wrote {out_path} ({len(cells)} cells)")
