"""
Unified diagnostics: S_transfer, Theorem 5.1 conditions, persist to file.
Per doc/stage7_unify.md Section 6. Aligns with Stage 5/6 outputs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(ROOT / config_path) as f:
        return yaml.safe_load(f)


def _to_json_serializable(obj: Any) -> Any:
    """Convert numpy types for json."""
    import numpy as np

    if isinstance(obj, (np.bool_, np.integer)):
        return bool(obj) if isinstance(obj, np.bool_) else int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return f if (f == f and abs(f) != float("inf")) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(x) for x in obj]
    return obj


def write_unified_diagnostics(
    ckpt_dir: Path,
    config_path: str = "config.yaml",
    stage6_results: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    Aggregate Stage 5 + Stage 6 into unified_diagnostics.json.
    Per doc/stage7_unify.md: S_transfer (formula 58), Theorem 5.1 three conditions.

    Args:
        ckpt_dir: Checkpoints directory.
        config_path: Config file path.
        stage6_results: Optional list from run_transfer_experiment; if None, load from stage6_experiment_results.json.

    Returns:
        Path to unified_diagnostics.json.
    """
    cfg = _load_config(config_path)
    xfer = cfg.get("transfer", {})
    rho_threshold = xfer.get("rho_regime_threshold", 0.5)
    weakness_threshold = xfer.get("weakness_consistency_threshold", 0.2)

    out_path = ckpt_dir / "unified_diagnostics.json"

    # Load Stage 5 report
    stage5_path = ckpt_dir / "stage5_transfer_report.json"
    if not stage5_path.exists():
        with open(out_path, "w") as f:
            json.dump({"error": "stage5_transfer_report.json not found"}, f, indent=2)
        return out_path

    with open(stage5_path) as f:
        stage5 = json.load(f)

    pairs = stage5.get("pairs", [])
    diagnostics: Dict[str, Any] = {
        "version": "stage7_unified",
        "source": "stage5_transfer_report",
        "pairs": [],
        "theorem_51": {},
    }

    for p in pairs:
        target = p.get("target", "")
        w_jd = p.get("w_jd_effective")
        w_crit = p.get("w_crit")
        rho = p.get("rho")
        delta_wpt = p.get("delta_wpt_p90")
        s_transfer = p.get("s_transfer")
        decision = p.get("decision", "")

        # Theorem 5.1 conditions
        cond1 = w_jd is not None and w_crit is not None and w_jd < w_crit
        cond2 = rho is not None and rho > rho_threshold
        cond3 = delta_wpt is not None and delta_wpt < weakness_threshold
        all_satisfied = cond1 and cond2 and cond3

        pair_diag = {
            "target": target,
            "s_transfer": s_transfer,
            "w_jd_effective": w_jd,
            "w_crit": w_crit,
            "rho": rho,
            "delta_wpt_p90": delta_wpt,
            "decision": decision,
            "theorem_51": {
                "statistical_compat": cond1,
                "structural_align": cond2,
                "behavioral_consist": cond3,
                "all_satisfied": all_satisfied,
            },
        }
        diagnostics["pairs"].append(_to_json_serializable(pair_diag))

    # Add stage6 metrics if available
    if stage6_results is None:
        stage6_path = ckpt_dir / "stage6_experiment_results.json"
        if stage6_path.exists():
            try:
                with open(stage6_path) as f:
                    stage6_results = json.load(f)
            except Exception:
                stage6_results = []

    if stage6_results:
        by_target = {r.get("target_asset", ""): r for r in stage6_results}
        for pd in diagnostics["pairs"]:
            t = pd.get("target", "")
            if t in by_target:
                sr = by_target[t]
                pd["scratch_nll"] = sr.get("scratch", {}).get("nll")
                pd["transweave_nll"] = sr.get("transweave", {}).get("nll") if "transweave" in sr else None
                pd["direct_nll"] = sr.get("direct", {}).get("nll") if "direct" in sr else None

    with open(out_path, "w") as f:
        json.dump(_to_json_serializable(diagnostics), f, indent=2)

    return out_path
