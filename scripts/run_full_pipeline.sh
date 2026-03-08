#!/usr/bin/env bash
#
# Full pipeline: lambda fix (empirical + static JD) -> unified 3,5,6 -> target TFT -> stage8a eval.
#
# One-click launch (recommended, survives screen sleep + disconnect):
#   caffeinate -i nohup bash scripts/run_full_pipeline.sh >> run_full_pipeline.log 2>&1 &
#   tail -f run_full_pipeline.log
#
# Or: ./scripts/run_full_pipeline_launch.sh
#
set -e
set -o pipefail  # pipeline fails if any command fails
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export ROOT
export PYTHONPATH="$ROOT:$PYTHONPATH"
LOG="${LOG:-run_full_pipeline.log}"
echo "===== run_full_pipeline started $(date) =====" | tee -a "$LOG"

# Step 0: Empirical + Static JD for all 4 assets (requires fix_lambda implementation)
python scripts/run_stage2_all_assets.py 2>&1 | tee -a "$LOG" || {
  echo "[FAIL] Stage 2 all assets. Ensure run_stage2_all_assets.py exists (see .claude/fix_lambda.md)" | tee -a "$LOG"
  exit 1
}
echo "[OK] Stage 2 all assets (empirical + static JD)" | tee -a "$LOG"

# Step 1: Unified 3,5,6 (skip stage 2 - already done above)
python scripts/run_unified.py --mode force --stage 3,5,6 2>&1 | tee -a "$LOG" || exit 1
echo "[OK] Unified stages 3,5,6" | tee -a "$LOG"

# Step 1.5: Stage 8B lambda ablation (eth_tft_jd_lambda0.ckpt + eth_static_jd_params_mle_unconstrained.json for report fig4)
python scripts/run_stage8b_lambda_ablation.py 2>&1 | tee -a "$LOG" || echo "[WARN] Stage 8b failed, report fig4 may be incomplete" | tee -a "$LOG"
echo "[OK] Stage 8b lambda ablation" | tee -a "$LOG"

# Step 2: Target TFT (BTC/SOL/DOGE, base + scratch + finetune)
python scripts/run_train_target_tft.py --force 2>&1 | tee -a "$LOG" || exit 1
echo "[OK] Target TFT training" | tee -a "$LOG"

# Step 3: Stage 8a evaluation
python scripts/run_stage8a_eval.py --skip-target-train 2>&1 | tee -a "$LOG" || exit 1
echo "[OK] Stage 8a eval" | tee -a "$LOG"

# Step 4: Generate report figures (non-fatal: some may need extra data)
python scripts/run_stage8a_figures.py 2>&1 | tee -a "$LOG" || echo "[WARN] Some figures may be placeholders" | tee -a "$LOG"
echo "[OK] Stage 8a figures" | tee -a "$LOG"

# Step 5: Write manifest for report (all outputs archived here)
CKPT="${ROOT}/experiments/checkpoints"
REPORT="${ROOT}/report"
python - "$ROOT" <<'PY' 2>&1 | tee -a "$LOG"
import json
import os
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("ROOT", "."))
ckpt = ROOT / "experiments" / "checkpoints"
report = ROOT / "report"
report.mkdir(parents=True, exist_ok=True)

# All key outputs for report writing
stage8a_jsons = [
    "stage8a_eth_ablation.json", "stage8a_target_tft.json", "stage8a_transfer_eval.json",
    "stage8a_s_correlation.json", "stage8a_theorem_verification.json", "stage8a_transweave_vs_target_tft.json",
]
other_jsons = ["stage5_transfer_report.json", "stage6_experiment_results.json"]

manifest = {
    "completed_at": datetime.now().isoformat(),
    "ready_for_report": True,
    "paths": {"checkpoints_dir": str(ckpt), "report_dir": str(report), "figures_dir": str(report / "figures")},
    "files_exist": {},
}
manifest["files_exist"]["stage8a_json"] = {n: (ckpt / n).exists() for n in stage8a_jsons}
manifest["files_exist"]["other_json"] = {n: (ckpt / n).exists() for n in other_jsons}

fig_dir = report / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
manifest["files_exist"]["figures"] = {f"fig{i}.png": (fig_dir / f"fig{i}.png").exists() for i in range(1, 11)}

with open(report / "run_full_pipeline_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# Human-readable quick start for report writing
md = ["# 报告撰写快速入口 (run_full_pipeline 产出)\n", f"完成时间: {manifest['completed_at']}\n"]
md.append("\n## 关键数据文件\n")
for cat in ["stage8a_json", "other_json"]:
    if cat not in manifest["files_exist"]:
        continue
    md.append(f"### {cat}\n")
    for n, ex in manifest["files_exist"].get(cat, {}).items():
        md.append(f"- `{ckpt / n}` {'✓' if ex else '✗'}\n")
md.append("\n## 图表 (report/figures/)\n")
for n, ex in manifest["files_exist"].get("figures", {}).items():
    md.append(f"- {n} {'✓' if ex else '✗'}\n")
md.append("\n## 更新 final_report.md 时\n")
md.append("- §3 Ablation: 填 stage8a_eth_ablation.json 的 NLL/CRPS\n")
md.append("- §4 Transfer: 填 stage8a_transweave_vs_target_tft.json 的 ΔNLL、grade\n")
md.append("- §4 S_transfer: stage8a_s_correlation.json 的 pearson_r\n")
with open(report / "REPORT_QUICK_START.md", "w") as f:
    f.writelines(md)
print("[OK] Manifest + REPORT_QUICK_START.md written")
PY

echo "===== run_full_pipeline done $(date) =====" | tee -a "$LOG"
echo "Report-ready manifest: report/run_full_pipeline_manifest.json" | tee -a "$LOG"
