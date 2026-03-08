"""
Unified framework: Algorithm 1 orchestration. Per doc/stage7_unify.md.
Aligns with Stage 6: transfer_map_{target}.pt, weak_model_{target}.pt.
Supports --mode skip/force and --stage for checkpoint caching.
Writes stage7_run_summary.json for post-run visualization (verify_stage7.ipynb).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch
import yaml

# Project root
ROOT = Path(__file__).resolve().parents[2]


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(ROOT / config_path) as f:
        return yaml.safe_load(f)


def _get_ckpt_dir(config: dict) -> Path:
    return ROOT / config.get("paths", {}).get("checkpoints", "experiments/checkpoints")


# --- Stage 2 cache ---
def _stage2_cache_exists(ckpt_dir: Path) -> bool:
    p = ckpt_dir / "eth_static_jd_params.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            json.load(f)
        return True
    except Exception:
        return False


def _run_stage2(config_path: str, ckpt_dir: Path) -> bool:
    """Run Stage 2: static MAP. Returns True if succeeded."""
    from src.models.jump_diffusion import load_eth_train_returns, fit_static_mle

    r_train = load_eth_train_returns(config_path)
    fit_static_mle(r_train, config_path, "eth_static_jd_params.json", asset="ETH")
    return True


# --- Stage 3 cache ---
# Stage 6 requires eth_tft_jd_w_finetune.ckpt; Stage 8 ablation needs eth_tft_jd_w_scratch.ckpt
def _stage3_cache_exists(ckpt_dir: Path) -> bool:
    return (
        (ckpt_dir / "eth_tft_jd.ckpt").exists()
        and (ckpt_dir / "eth_tft_jd_w_scratch.ckpt").exists()
        and (ckpt_dir / "eth_tft_jd_w_finetune.ckpt").exists()
    )


def _run_train_tft_jd(args: list[str]) -> int:
    """Helper for subprocess: run train_tft_jd_eth.py with given args. Returns returncode."""
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_tft_jd_eth.py")] + args,
        cwd=str(ROOT),
        capture_output=False,
    ).returncode


def _run_stage3(config_path: str, force: bool = False) -> bool:
    """
    Run Stage 3: train ETH TFT-JD (base + scratch + finetune).
    Layout: 1-2 = step1 base, step2 scratch+finetune in parallel.
    scratch and finetune are independent of each other; finetune needs base.

    When force=True, run all training regardless of existing checkpoints (overwrites).
    """
    train_script = str(ROOT / "scripts" / "train_tft_jd_eth.py")
    ckpt_dir = ROOT / _load_config(config_path).get("paths", {}).get("checkpoints", "experiments/checkpoints")

    # Step 1: eth_tft_jd.ckpt (base, no weakness) - required before finetune
    if force or not (ckpt_dir / "eth_tft_jd.ckpt").exists():
        print("[Stage 3] Training eth_tft_jd.ckpt (base)...")
        if subprocess.run([sys.executable, train_script], cwd=str(ROOT), capture_output=False).returncode != 0:
            return False

    # Step 2: run scratch and finetune in parallel (both only need base to exist for finetune)
    to_run: list[tuple[str, list[str]]] = []
    if force or not (ckpt_dir / "eth_tft_jd_w_scratch.ckpt").exists():
        to_run.append(("eth_tft_jd_w_scratch.ckpt", ["--use_weakness", "true", "--mode", "scratch"]))
    if force or not (ckpt_dir / "eth_tft_jd_w_finetune.ckpt").exists():
        to_run.append(("eth_tft_jd_w_finetune.ckpt", ["--use_weakness", "true", "--mode", "finetune"]))

    if not to_run:
        return True

    if len(to_run) == 1:
        name, args = to_run[0]
        print(f"[Stage 3] Training {name}...")
        return subprocess.run(
            [sys.executable, train_script] + args,
            cwd=str(ROOT),
            capture_output=False,
        ).returncode == 0

    # Parallel or sequential: run scratch and finetune
    use_parallel = _load_config(config_path).get("unified", {}).get("stage3_parallel", True)
    if use_parallel:
        print("[Stage 3] Training eth_tft_jd_w_scratch.ckpt and eth_tft_jd_w_finetune.ckpt in parallel...")
        with ProcessPoolExecutor(max_workers=2) as ex:
            futures = {ex.submit(_run_train_tft_jd, args): name for name, args in to_run}
            for fut in as_completed(futures):
                name = futures[fut]
                rc = fut.result()
                if rc != 0:
                    print(f"[Stage 3] {name} failed (returncode={rc})")
                    return False
    else:
        for name, args in to_run:
            print(f"[Stage 3] Training {name}...")
            if subprocess.run([sys.executable, train_script] + args, cwd=str(ROOT), capture_output=False).returncode != 0:
                return False
    return True


# --- Stage 5 cache ---
def _stage5_cache_exists(ckpt_dir: Path) -> bool:
    p = ckpt_dir / "stage5_transfer_report.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get("version") == "v3_paper_formula"
    except Exception:
        return False


def _run_stage5(config_path: str) -> bool:
    """Run Stage 5: regime + transfer metrics."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_stage5.py")],
        cwd=str(ROOT),
        capture_output=False,
    )
    return result.returncode == 0


# --- Stage 6 cache (per target, align with Stage 6) ---
def _stage6_cache_exists(ckpt_dir: Path, target: str) -> bool:
    """Stage 6 outputs transfer_map_{target}.pt per doc/stage6_transfer.md."""
    return (ckpt_dir / f"transfer_map_{target}.pt").exists()


def _run_stage6(
    config_path: str,
    ckpt_dir: Path,
    target_assets: List[str],
    mode: str,
) -> List[Dict[str, Any]]:
    """
    Run Stage 6 for each target. Skip targets with existing transfer_map_{target}.pt when mode=skip.
    Returns list of results from run_transfer_experiment.
    """
    from src.data.tft_dataset import get_split_indices, load_feature_arrays
    from src.models.tft_jd import build_tft_jd, infer_eth_theta
    from src.transfer.transweave import run_transfer_experiment

    cfg = _load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ETH TFT-JD+W finetune (Stage 6 requires this per doc/stage6_transfer.md 18.4)
    eth_ckpt = ckpt_dir / "eth_tft_jd_w_finetune.ckpt"
    if not eth_ckpt.exists():
        raise FileNotFoundError(
            f"ETH checkpoint not found: {eth_ckpt}. "
            "Run Stage 3 (train_tft_jd_eth.py --use_weakness true --mode finetune) first."
        )
    model = build_tft_jd()
    ck = torch.load(eth_ckpt, map_location=device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.to(device)

    # Infer ETH theta
    X_hist, Z_future, _, split_arr = load_feature_arrays("ETH", config_path)
    eth_split_indices = get_split_indices(split_arr)
    theta_eth_15min = infer_eth_theta(
        model, X_hist, Z_future,
        batch_size=cfg.get("transfer_map", {}).get("batch_size", 2048),
        device=device,
    )

    # Load Stage 5 report
    with open(ckpt_dir / "stage5_transfer_report.json") as f:
        stage5_report = json.load(f)
    assert stage5_report.get("version") == "v3_paper_formula"

    all_results: List[Dict[str, Any]] = []
    existing_by_target: Dict[str, dict] = {}
    stage6_path = ckpt_dir / "stage6_experiment_results.json"
    if stage6_path.exists():
        try:
            with open(stage6_path) as f:
                for r in json.load(f):
                    existing_by_target[r.get("target_asset", "")] = r
        except Exception:
            pass

    for target in target_assets:
        if mode == "skip" and _stage6_cache_exists(ckpt_dir, target):
            print(f"[Stage 6] Skipping {target} (transfer_map_{target}.pt exists)")
            if target in existing_by_target:
                all_results.append(existing_by_target[target])
            continue
        try:
            results = run_transfer_experiment(
                target_asset=target,
                stage5_report=stage5_report,
                theta_eth_15min=theta_eth_15min,
                eth_split_indices=eth_split_indices,
                config=cfg,
                config_path=config_path,
                ckpt_dir=ckpt_dir,
            )
            all_results.append(results)
        except Exception as e:
            print(f"[{target}] Failed: {e}")
            import traceback
            traceback.print_exc()

    # Save stage6_experiment_results.json (append/merge if partial skip)
    def _to_json_serializable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return bool(obj) if isinstance(obj, np.bool_) else int(obj)
        if isinstance(obj, np.floating):
            f = float(obj)
            return f if np.isfinite(f) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json_serializable(x) for x in obj]
        return obj

    with open(ckpt_dir / "stage6_experiment_results.json", "w") as f:
        json.dump([_to_json_serializable(r) for r in all_results], f, indent=2)

    return all_results


def run_algorithm_1(
    source_asset: str = "ETH",
    target_assets: Optional[List[str]] = None,
    mode: str = "skip",
    stages: Optional[List[int]] = None,
    config_path: str = "config.yaml",
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Execute Algorithm 1 end-to-end. Per doc/stage7_unify.md.

    Args:
        source_asset: Source asset (default ETH).
        target_assets: Target assets (default BTC, SOL, DOGE).
        mode: "skip" = use cache if exists; "force" = always run.
        stages: List of stages to run (default [2,3,5,6]). E.g. [5,6] for 5 and 6 only.
        config_path: Config file path.
        config: Override config dict (if None, load from config_path).

    Returns:
        Summary dict with stage_status, checkpoints, diagnostics path.
    """
    target_assets = target_assets or ["BTC", "SOL", "DOGE"]
    stages = stages or [2, 3, 5, 6]
    cfg = config or _load_config(config_path)
    ckpt_dir = _get_ckpt_dir(cfg)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "stage_status": {},
        "checkpoints": {},
        "stage6_results_count": 0,
    }

    # Stage 2
    if 2 in stages:
        if mode == "skip" and _stage2_cache_exists(ckpt_dir):
            summary["stage_status"]["2"] = "skipped"
        else:
            print("[Stage 2] Running static MAP...")
            _run_stage2(config_path, ckpt_dir)
            summary["stage_status"]["2"] = "run"

    # Stage 3
    if 3 in stages:
        if mode == "skip" and _stage3_cache_exists(ckpt_dir):
            summary["stage_status"]["3"] = "skipped"
        else:
            print("[Stage 3] Training ETH TFT-JD...")
            ok = _run_stage3(config_path, force=(mode == "force"))
            summary["stage_status"]["3"] = "run" if ok else "failed"

    # Stage 5
    if 5 in stages:
        if mode == "skip" and _stage5_cache_exists(ckpt_dir):
            summary["stage_status"]["5"] = "skipped"
        else:
            print("[Stage 5] Running regime + transfer metrics...")
            _run_stage5(config_path)
            summary["stage_status"]["5"] = "run"

    # Stage 6
    if 6 in stages:
        print("[Stage 6] Running TransWeave transfer...")
        stage6_results = _run_stage6(config_path, ckpt_dir, target_assets, mode)
        summary["stage_status"]["6"] = "run"
        summary["stage6_results_count"] = len(stage6_results)

        # Write unified diagnostics from stage5 + stage6
        from .diagnostics import write_unified_diagnostics

        diag_path = write_unified_diagnostics(
            ckpt_dir, config_path, stage6_results=stage6_results
        )
        summary["diagnostics_path"] = str(diag_path)

    # Write stage7_run_summary.json for post-run visualization (verify_stage7.ipynb)
    summary["run_timestamp"] = datetime.now().isoformat()
    summary["stages_run"] = list(stages)
    summary["mode"] = mode
    summary["output_paths"] = {
        "stage2": str(ckpt_dir / "eth_static_jd_params.json"),
        "stage3_ckpts": [
            str(ckpt_dir / "eth_tft_jd.ckpt"),
            str(ckpt_dir / "eth_tft_jd_w_scratch.ckpt"),
            str(ckpt_dir / "eth_tft_jd_w_finetune.ckpt"),
        ],
        "stage5": str(ckpt_dir / "stage5_transfer_report.json"),
        "stage6": str(ckpt_dir / "stage6_experiment_results.json"),
        "diagnostics": str(ckpt_dir / "unified_diagnostics.json"),
    }
    # Per-target summary for quick viz
    if summary.get("stage6_results_count", 0) > 0:
        stage6_path = ckpt_dir / "stage6_experiment_results.json"
        if stage6_path.exists():
            try:
                with open(stage6_path) as f:
                    s6 = json.load(f)
                summary["stage6_per_target"] = [
                    {
                        "target": r.get("target_asset", ""),
                        "decision": r.get("decision", ""),
                        "scratch_nll": r.get("scratch", {}).get("nll"),
                        "transweave_nll": r.get("transweave", {}).get("nll") if "transweave" in r else None,
                        "direct_nll": r.get("direct", {}).get("nll"),
                        "phase3_epochs": len(r.get("phase3_history", [])),
                        "phase4_epochs": len(r.get("phase4_history", [])),
                    }
                    for r in s6
                ]
            except Exception:
                pass
    summary_path = ckpt_dir / "stage7_run_summary.json"

    def _to_json_serializable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return bool(obj) if isinstance(obj, np.bool_) else int(obj)
        if isinstance(obj, np.floating):
            f = float(obj)
            return f if np.isfinite(f) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_json_serializable(x) for x in obj]
        return obj

    with open(summary_path, "w") as f:
        json.dump(_to_json_serializable(summary), f, indent=2)
    summary["run_summary_path"] = str(summary_path)

    return summary
