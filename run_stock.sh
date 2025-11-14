#!/usr/bin/env bash
set -euo pipefail

# run_stock.sh
# Activates the preferred virtualenv (data-volume venv preferred),
# sets pip/TMPDIR to avoid root disk use, and runs StockAnalysis.py

VENV_DATA="/mnt/data/cron-analysis/venv"
VENV_ROOT="/root/cron-n-analysis/venv"

if [ -d "$VENV_DATA" ]; then
  VENV="$VENV_DATA"
elif [ -d "$VENV_ROOT" ]; then
  VENV="$VENV_ROOT"
else
  echo "No virtualenv found at $VENV_DATA or $VENV_ROOT" >&2
  exit 1
fi

echo "Activating venv: $VENV"
. "$VENV/bin/activate"

export TMPDIR=/mnt/data/tmp
mkdir -p "$TMPDIR"

cd "$(dirname "$0")"

echo "Running StockAnalysis.py in $(pwd) with Python: $(which python)"
python StockAnalysis.py
