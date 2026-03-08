# TransWeave-Finance

Reproduction of **"A Unified Framework for Jump-Diffusion Modeling and Transfer Learning in Financial Markets with Behavioral Risk Measures"** (Goertzel, 2025).

**One-line quick start:** `pip install -r requirements.txt` then `python scripts/run_unified.py --mode skip && python scripts/run_stage8a_eval.py --skip-target-train` (~minutes with cached data).

We implement the full TransWeave pipeline: Merton jump-diffusion with TFT neural conditioning, prospect-theoretic weakness regularization, HMM regime detection, and parameter-space transfer learning. ETH serves as the source asset; BTC, SOL, and DOGE are transfer targets.

## Results

### ETH Source Model Ablation

| Model | NLL | CRPS | VaR 5% Breach |
|-------|-----|------|---------------|
| GBM | -4.115 | 0.00199 | — |
| Static-JD | -4.156 | 0.00198 | 9.1% |
| **TFT-JD** | **-4.343** | **0.00191** | 5.6% |
| TFT-JD+W | -4.312 | 0.00192 | 6.7% |

### Transfer Learning

| Target | Decision | ρ | S_transfer | Scratch NLL | TransWeave NLL | Target-TFT NLL | Grade |
|--------|----------|-------|-----------|-------------|----------------|-----------------|-------|
| BTC | Full | 0.768 | 0.570 | -4.788 | **-4.860** | -4.888 | Good |
| SOL | Partial | 0.714 | 0.391 | -4.177 | -4.142 | -4.211 | Acceptable |
| DOGE | Partial | 0.662 | 0.440 | -4.038 | **-4.084** | -4.083 | Excellent |

TransWeave improves over Scratch for BTC and DOGE; the composite transfer score $S_{\text{transfer}}$ perfectly ranks asset transferability (Spearman r = -1.0).

## Quick Start

**Environment**: Python 3.10+, PyTorch 2.x. We recommend conda:

```bash
conda create -n transweave python=3.10
conda activate transweave
pip install -r requirements.txt
```

**Quick evaluation** (uses cached checkpoints, ~minutes):

```bash
python scripts/run_unified.py --mode skip
python scripts/run_stage8a_eval.py --skip-target-train
```

**Full retraining** (~9 hours on Apple M-series):

```bash
bash scripts/run_full_pipeline.sh
```

This runs Stage 2 → Unified 3,5,6 → **Stage 8B** (λ ablation) → Target TFT → Stage 8a eval → figures. Stage 8B produces `eth_tft_jd_lambda0.ckpt` and `eth_static_jd_params_mle_unconstrained.json` required for the report's Figure 4.

**Generate report** (Jupyter notebook with hidden code cells):

```bash
cd report && jupyter nbconvert --to pdf --no-input reproduction_report.ipynb
```

## Data

Pre-computed features (`data/features/*.npz`) and model checkpoints (`experiments/checkpoints/`) are included. To rebuild data from scratch (requires ccxt + Dune API access):

```bash
python scripts/run_stage1.py        # fetch & preprocess OHLCV + on-chain
bash scripts/run_full_pipeline.sh   # full pipeline: Stage 2 → 3 → 5 → 6 → 8a
```

## Project Structure

```
transweave-finance/
├── config.yaml                  # All hyperparameters (no hardcoded values)
├── requirements.txt
├── README.md
│
├── src/
│   ├── data/                    # Stage 1: fetch, preprocess, feature engineering
│   │   ├── fetcher.py           # OHLCV via ccxt
│   │   ├── preprocessor.py      # Log returns (Eq. 1)
│   │   ├── features.py          # Feature engineering (Eq. 2)
│   │   └── tft_dataset.py       # TFT input format
│   ├── models/                  # Stage 2-3: JD model + TFT conditioning
│   │   ├── jump_diffusion.py    # JD density (Eq. 1, 8), static MAP
│   │   ├── tft.py               # TFT encoder (Eq. 2)
│   │   ├── tft_jd.py            # Joint TFT-JD (Eq. 3-6)
│   │   └── losses.py            # NLL + CRPS (Eq. 7)
│   ├── behavioral/              # Stage 4: prospect theory
│   │   ├── prospect.py          # Prelec weighting, value function (Eq. 35)
│   │   ├── weakness.py          # W_PT measure (Eq. 36-37)
│   │   └── regularizer.py       # Weakness regularizer (Eq. 43-44, 50)
│   ├── transfer/                # Stage 5-6: regime detection + TransWeave
│   │   ├── regime.py            # HMM regime (Eq. 13)
│   │   ├── metrics.py           # W_JD, ρ, S_transfer (Eq. 12-13, 58)
│   │   ├── transweave.py        # Transfer map T (Eq. 9-11, 49), Algorithm 1
│   │   └── feature_shift.py     # Feature shift D_H (Def. 3.1)
│   ├── unified/                 # Stage 7: end-to-end pipeline
│   │   ├── framework.py         # run_algorithm_1()
│   │   └── diagnostics.py       # Unified diagnostics (Eq. 58)
│   └── evaluation/              # Stage 8: evaluation
│       ├── calibration.py       # PIT, KS test (Appendix A.1)
│       ├── tail_risk.py         # VaR backtest (Appendix A.2)
│       ├── transfer_eval.py     # S_transfer vs NLL, Theorem 5.1
│       └── empirical_jump.py    # BNS + Lee-Mykland estimators
│
├── scripts/
│   ├── run_full_pipeline.sh     # One-click: Stage 2 → 8a (~9h)
│   ├── run_unified.py           # Algorithm 1: --mode skip|force
│   ├── run_stage2_all_assets.py # Static JD for all assets
│   ├── train_tft_jd_eth.py      # ETH TFT-JD training
│   ├── run_train_target_tft.py  # Target asset TFT training
│   ├── run_stage8a_eval.py      # Evaluation metrics
│   └── run_stage8a_figures.py   # Generate report figures
│
├── report/
│   ├── reproduction_report.ipynb  # Full reproduction report (→ PDF)
│   └── figures/                   # Generated figures
│
├── notebooks/                   # Development & verification notebooks
│   ├── verify_static_jd.ipynb
│   ├── verify_tft_jd.ipynb
│   ├── verify_weakness.ipynb
│   ├── verify_regime.ipynb
│   ├── verify_transfer.ipynb
│   └── verify_stage7.ipynb
│
├── data/features/               # Pre-computed TFT arrays (.npz)
└── experiments/checkpoints/     # Model checkpoints + result JSONs
```

## Configuration

All hyperparameters live in `config.yaml`, organized by stage:

| Section | Stage | Key Parameters |
|---------|-------|----------------|
| `stage1` | Data | assets, train/val/test splits, window size |
| `static_jd_map` | Stage 2 | p_center, prior_weight |
| `training` | Stage 2-3 | lr, epochs, batch_size, jd_truncation_n |
| `tft_jd` | Stage 3 | hidden_size, dropout, lambda_prior_weight |
| `prospect_theory` | Stage 4 | delta, alpha, beta, lambda_loss |
| `transfer` | Stage 5 | hmm_n_states, rho_full/partial, rolling_window |
| `transfer_map` | Stage 6 | lambda_tw, phase3/4 lr, patience |

## Training Time

Full pipeline (`run_full_pipeline.sh`): approximately **9 hours** on Apple M-series, broken down as:
- ETH TFT-JD training (3 modes): ~2.5h
- Target TFT training (3 assets × 3 modes): ~5h
- Stage 2/5/6/8a: ~1.5h

## License

Research reproduction for academic purposes.
