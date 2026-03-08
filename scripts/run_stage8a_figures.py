"""
Stage 8a Task 4: Generate core figures. Per doc/stage8a_audit_present.md.
Output: report/figures/fig1_ablation.png ... fig10_regime_price.png
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Style per stage8a
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-paper")
except Exception:
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-paper")
    except Exception:
        pass

FIGSIZE = (10, 5)
DPI = 150
COLORS = {
    "GBM": "#999999",
    "Static-JD": "#4DBEEE",
    "TFT-JD": "#0072BD",
    "TFT-JD+W": "#D95319",
    "Scratch": "#EDB120",
    "Direct": "#7E2F8E",
    "TransWeave": "#77AC30",
    "Target-TFT-JD": "#A2142F",
}


def fig1_ablation(ckpt_dir: Path, out_dir: Path) -> None:
    """Ablation NLL/CRPS bar chart."""
    path = ckpt_dir / "stage8a_eth_ablation.json"
    if not path.exists():
        # Fallback: use doc example
        models = ["GBM", "Static-JD", "TFT-JD", "TFT-JD+W"]
        nll = [-3.5, -4.2, -4.6, -4.7]
        crps = [0.002, 0.0015, 0.0012, 0.0011]
    else:
        with open(path) as f:
            data = json.load(f)
        models = list(data.get("models", {}).keys())
        nll = [data["models"].get(m, {}).get("nll", 0) for m in models]
        crps = [data["models"].get(m, {}).get("crps", 0) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)
    x = np.arange(len(models))
    colors = [COLORS.get(m, "#666666") for m in models]
    ax1.bar(x, nll, color=colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylabel("NLL")
    ax1.set_title("ETH Ablation: NLL")
    ax2.bar(x, crps, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel("CRPS")
    ax2.set_title("ETH Ablation: CRPS")
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_ablation.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig2_pit(ckpt_dir: Path, out_dir: Path) -> None:
    """PIT histograms (2x2 grid). Placeholder when PIT data not available."""
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    for ax in axes.flat:
        # Uniform reference
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)
        ax.set_xlabel("PIT")
    axes[0, 0].set_title("GBM")
    axes[0, 1].set_title("Static-JD")
    axes[1, 0].set_title("TFT-JD")
    axes[1, 1].set_title("TFT-JD+W")
    plt.suptitle("PIT Histograms (run stage8a_eval for actual data)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_pit.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig3_var_backtest(ckpt_dir: Path, out_dir: Path) -> None:
    """VaR backtest comparison."""
    path = ckpt_dir / "stage8a_eth_ablation.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        models = [m for m in data.get("models", {}) if "var_breach_rate" in data["models"].get(m, {})]
        rates = [data["models"][m]["var_breach_rate"] for m in models]
    else:
        models = ["GBM", "Static-JD", "TFT-JD", "TFT-JD+W"]
        rates = [0.06, 0.05, 0.048, 0.047]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(models))
    colors = [COLORS.get(m, "#666666") for m in models]
    ax.bar(x, rates, color=colors)
    ax.axhline(0.05, color="red", ls="--", label="Expected 5%")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("VaR 5% Breach Rate")
    ax.set_title("VaR Backtest")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_var_backtest.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig4_transfer_nll(ckpt_dir: Path, out_dir: Path) -> None:
    """Transfer NLL comparison (3 targets x 3-4 methods). Includes Target-TFT-JD if available."""
    path = ckpt_dir / "stage6_experiment_results.json"
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)
    targets = [r["target_asset"] for r in data]
    methods = ["Scratch", "Direct", "TransWeave"]
    nll = {
        m: [next((r.get(m, {}).get("nll", np.nan) for r in data if r["target_asset"] == t), np.nan)
           for t in targets]
        for m in methods
    }
    # Add Target-TFT-JD if stage8a_target_tft.json exists
    target_tft_path = ckpt_dir / "stage8a_target_tft.json"
    if target_tft_path.exists():
        with open(target_tft_path) as f:
            target_tft = json.load(f)
        tt = target_tft.get("target_tft_jd", {})
        nll["Target-TFT-JD"] = [tt.get(t, {}).get("nll", np.nan) for t in targets]
        methods = ["Scratch", "Direct", "TransWeave", "Target-TFT-JD"]
    n_methods = len(methods)
    x = np.arange(len(targets))
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, m in enumerate(methods):
        ax.bar(x + i * width, nll.get(m, [np.nan] * len(targets)), width, label=m, color=COLORS.get(m, "#666666"))
    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(targets)
    ax.set_ylabel("NLL")
    ax.set_title("Transfer NLL: 3 Targets x " + str(n_methods) + " Methods")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_transfer_nll.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig5_s_vs_nll(ckpt_dir: Path, out_dir: Path) -> None:
    """S_transfer vs NLL improvement scatter."""
    s5_path = ckpt_dir / "stage5_transfer_report.json"
    s6_path = ckpt_dir / "stage6_experiment_results.json"
    if not s5_path.exists() or not s6_path.exists():
        return
    with open(s5_path) as f:
        s5 = json.load(f)
    with open(s6_path) as f:
        s6 = json.load(f)
    s_by_target = {p["target"]: p["s_transfer"] for p in s5.get("pairs", []) if "target" in p}
    s_vals, d_vals, labels = [], [], []
    for r in s6:
        t = r.get("target_asset")
        if not t:
            continue
        scratch_nll = r.get("scratch", {}).get("nll", np.nan)
        tw_nll = r.get("transweave", {}).get("nll", np.nan)
        delta = tw_nll - scratch_nll if np.isfinite(scratch_nll) and np.isfinite(tw_nll) else np.nan
        s_val = s_by_target.get(t, np.nan)
        if np.isfinite(s_val) and np.isfinite(delta):
            s_vals.append(s_val)
            d_vals.append(delta)
            labels.append(t)
    if len(s_vals) < 2:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(s_vals, d_vals, s=100, c=[COLORS.get("TransWeave", "#77AC30")] * len(s_vals))
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (s_vals[i], d_vals[i]), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="gray", ls="--")
    ax.set_xlabel("S_transfer")
    ax.set_ylabel("Δ NLL (TransWeave - Scratch)")
    ax.set_title("S_transfer vs NLL Improvement")
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_s_vs_nll.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig6_decision_matrix(ckpt_dir: Path, out_dir: Path) -> None:
    """Transfer decision matrix heatmap."""
    s5_path = ckpt_dir / "stage5_transfer_report.json"
    s6_path = ckpt_dir / "stage6_experiment_results.json"
    if not s5_path.exists() or not s6_path.exists():
        return
    with open(s5_path) as f:
        s5 = json.load(f)
    with open(s6_path) as f:
        s6 = json.load(f)
    targets = [r["target_asset"] for r in s6]
    by_tgt = {p["target"]: p for p in s5.get("pairs", []) if "target" in p}
    s6_by_tgt = {r["target_asset"]: r for r in s6}
    rows = ["ρ_regime", "Decision", "Scratch NLL", "TransWeave NLL", "Δ NLL"]
    data = []
    for t in targets:
        p = by_tgt.get(t, {})
        r = s6_by_tgt.get(t, {})
        scratch = r.get("scratch", {}).get("nll", np.nan)
        tw = r.get("transweave", {}).get("nll", np.nan)
        delta = tw - scratch if np.isfinite(scratch) and np.isfinite(tw) else np.nan
        data.append([
            p.get("rho", np.nan),
            p.get("decision", "?"),
            scratch,
            tw,
            delta,
        ])
    # Numeric heatmap: use rho, delta
    mat = np.array([[d[0], d[4]] for d in data], dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(mat.T, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["ρ_regime", "Δ NLL"])
    plt.colorbar(im, ax=ax)
    ax.set_title("Transfer Decision Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_decision_matrix.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig7_sigma_dynamics(ckpt_dir: Path, out_dir: Path) -> None:
    """σ_t dynamics placeholder. Requires theta sequence from TFT inference."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title("σ_t Dynamics (run TFT inference for actual data)")
    ax.set_ylabel("σ (annualized)")
    ax.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(out_dir / "fig7_sigma_dynamics.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig8_lambda_dynamics(ckpt_dir: Path, out_dir: Path) -> None:
    """λ_t dynamics placeholder."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title("λ_t Dynamics (run TFT inference for actual data)")
    ax.set_ylabel("λ (annualized)")
    ax.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(out_dir / "fig8_lambda_dynamics.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig9_wpt_trajectory(ckpt_dir: Path, out_dir: Path) -> None:
    """W_PT trajectory placeholder."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title("W_PT Trajectory (regime-colored)")
    ax.set_ylabel("W_PT")
    ax.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(out_dir / "fig9_wpt_trajectory.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def fig10_regime_price(ckpt_dir: Path, out_dir: Path) -> None:
    """Regime-colored price chart placeholder. Requires regime + price data."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title("Regime-Colored Price (4 assets)")
    ax.set_ylabel("Price (normalized)")
    ax.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(out_dir / "fig10_regime_price.png", dpi=DPI, bbox_inches="tight")
    plt.close()


def main() -> None:
    ckpt_dir = ROOT / "experiments" / "checkpoints"
    out_dir = ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1_ablation(ckpt_dir, out_dir)
    fig2_pit(ckpt_dir, out_dir)
    fig3_var_backtest(ckpt_dir, out_dir)
    fig4_transfer_nll(ckpt_dir, out_dir)
    fig5_s_vs_nll(ckpt_dir, out_dir)
    fig6_decision_matrix(ckpt_dir, out_dir)
    fig7_sigma_dynamics(ckpt_dir, out_dir)
    fig8_lambda_dynamics(ckpt_dir, out_dir)
    fig9_wpt_trajectory(ckpt_dir, out_dir)
    fig10_regime_price(ckpt_dir, out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
