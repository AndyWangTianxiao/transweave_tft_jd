#!/usr/bin/env bash
#
# Quick evaluation using cached checkpoints (~minutes).
# Runs run_unified (skip mode) + run_stage8a_eval.
#
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PYTHONPATH"

python scripts/run_unified.py --mode skip
python scripts/run_stage8a_eval.py --skip-target-train
