# TransWeave-Finance

Reproduction of **"A Unified Framework for Jump-Diffusion Modeling and Transfer Learning in Financial Markets with Behavioral Risk Measures"** (Goertzel, 2025).

A practical implementation of the paper’s pipeline: Merton jump-diffusion with TFT conditioning, prospect-theoretic weakness regularization, HMM regime detection, and parameter-space transfer learning. Source asset: ETH. Targets: BTC, SOL, DOGE. See `report/reproduction_report.ipynb` for methodology and results.

**Quick start:** `pip install -r requirements.txt` then `bash scripts/run_quick_eval.sh` (minutes with cached checkpoints).

## Quick Start

**Environment**: Python 3.10+, PyTorch 2.x. We recommend conda:

```bash
conda create -n transweave python=3.10
conda activate transweave
pip install -r requirements.txt
```

**Quick evaluation** (uses cached checkpoints, ~minutes):

```bash
bash scripts/run_quick_eval.sh
```

**Full pipeline** (~9 hours on Apple M-series):

```bash
bash scripts/run_full_pipeline.sh
```

**Export report to PDF**:

```bash
cd report && jupyter nbconvert --to pdf --no-input reproduction_report.ipynb
```

## Data

Pre-computed features (`data/features/*.npz`), model checkpoints (`experiments/checkpoints/`), and onchain parquet (`data/processed/onchain/*.parquet`, ~4.5MB) are included.

To rebuild from scratch:
- **With onchain cached** (default): Stage 1 fetches OHLCV via ccxt only; skips Dune if `data/processed/onchain/` exists.
- **Full rebuild**: `DUNE_API_KEY` + `python scripts/run_stage1.py --force-onchain` to re-fetch onchain from Dune.

```bash
python scripts/run_stage1.py
bash scripts/run_full_pipeline.sh
```

## Project Structure

```
transweave-finance/
├── config.yaml                  # All hyperparameters (no hardcoded values)
├── requirements.txt
├── README.md
│
├── src/
│   ├── data/                    # fetch, preprocess, feature engineering
│   │   ├── fetcher.py           # OHLCV via ccxt
│   │   ├── preprocessor.py      # Log returns (Eq. 1)
│   │   ├── features.py          # Feature engineering (Eq. 2)
│   │   └── tft_dataset.py       # TFT input format
│   ├── models/                  # JD model + TFT conditioning
│   │   ├── jump_diffusion.py    # JD density (Eq. 1, 8), static MAP
│   │   ├── tft.py               # TFT encoder (Eq. 2)
│   │   ├── tft_jd.py            # Joint TFT-JD (Eq. 3-6)
│   │   └── losses.py            # NLL + CRPS (Eq. 7)
│   ├── behavioral/              # prospect theory
│   │   ├── prospect.py          # Prelec weighting, value function (Eq. 35)
│   │   ├── weakness.py          # W_PT measure (Eq. 36-37)
│   │   └── regularizer.py       # Weakness regularizer (Eq. 43-44, 50)
│   ├── transfer/                # regime detection + TransWeave
│   │   ├── regime.py            # HMM regime (Eq. 13)
│   │   ├── metrics.py           # W_JD, ρ, S_transfer (Eq. 12-13, 58)
│   │   ├── transweave.py        # Transfer map T (Eq. 9-11, 49), Algorithm 1
│   │   └── feature_shift.py     # Feature shift D_H (Def. 3.1)
│   ├── unified/                 # end-to-end pipeline
│   │   ├── framework.py         # run_algorithm_1()
│   │   └── diagnostics.py       # Unified diagnostics (Eq. 58)
│   └── evaluation/              # evaluation
│       ├── calibration.py       # PIT, KS test (Appendix A.1)
│       ├── tail_risk.py         # VaR backtest (Appendix A.2)
│       ├── transfer_eval.py     # S_transfer vs NLL, Theorem 5.1
│       └── empirical_jump.py    # BNS + Lee-Mykland estimators
│
├── scripts/
│   ├── run_full_pipeline.sh     # full pipeline (~9h)
│   ├── run_quick_eval.sh        # quick eval (cached checkpoints)
│   ├── run_unified.py           # Algorithm 1: --mode skip|force
│   ├── run_stage2_all_assets.py # Static JD for all assets
│   ├── train_tft_jd_eth.py      # ETH TFT-JD training
│   ├── run_train_target_tft.py  # Target asset TFT training
│   ├── run_stage8a_eval.py      # Evaluation metrics
│   └── run_stage8a_figures.py   # Generate report figures
│
├── report/
│   ├── reproduction_report.ipynb  # Methodology and results (→ PDF)
│   └── figures/                   # Generated figures
│
├── data/features/               # Pre-computed TFT arrays (.npz)
└── experiments/checkpoints/     # Model checkpoints + result JSONs
```

## Configuration

All hyperparameters live in `config.yaml`:

| Section | Key Parameters |
|---------|----------------|
| `stage1` | assets, train/val/test splits, window size |
| `static_jd_map` | p_center, prior_weight |
| `training` | lr, epochs, batch_size, jd_truncation_n |
| `tft_jd` | hidden_size, dropout, lambda_prior_weight |
| `prospect_theory` | delta, alpha, beta, lambda_loss |
| `transfer` | hmm_n_states, rho_full/partial, rolling_window |
| `transfer_map` | lambda_tw, phase3/4 lr, patience |

## Training Time

Full pipeline: approximately **9 hours** on Apple M-series. Quick evaluation: **minutes**.

## License

Research reproduction for academic purposes.
