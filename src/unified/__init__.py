"""
Unified framework: Algorithm 1 orchestration, diagnostics.
Per doc/stage7_unify.md. Aligns with Stage 6 outputs (transfer_map_*.pt, weak_model_*.pt).
"""

from .framework import run_algorithm_1
from .diagnostics import write_unified_diagnostics

__all__ = ["run_algorithm_1", "write_unified_diagnostics"]
