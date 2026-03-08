"""
Stage 8a evaluation module. Per doc/stage8a_audit_present.md Task 2.
Calibration (PIT, NLL), tail risk (VaR, CVaR), transfer evaluation.
"""

from .calibration import compute_pit, pit_uniformity_test, nll_per_sample
from .tail_risk import var_backtest, cvar_accuracy
from .transfer_eval import s_transfer_vs_nll, theorem_5_1_verification
from .gbm_baseline import evaluate_gbm_baseline

__all__ = [
    "compute_pit",
    "pit_uniformity_test",
    "nll_per_sample",
    "var_backtest",
    "cvar_accuracy",
    "s_transfer_vs_nll",
    "theorem_5_1_verification",
    "evaluate_gbm_baseline",
]
