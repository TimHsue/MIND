#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mode="${1:-auto}"
case "$mode" in -h|--help) echo "Usage: bash setup.sh [--cpu|--cuda]"; exit 0;; auto|--cpu|--cuda) ;; *) echo "Usage: bash setup.sh [--cpu|--cuda]"; exit 2;; esac
python="${PYTHON:-python3}"
"$python" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "Use Python 3.11 or 3.12"'
"$python" -m venv .venv
python="$PWD/.venv/bin/python"
"$python" -m pip install --upgrade pip
index=https://download.pytorch.org/whl/cpu
if [[ "$mode" == --cuda ]] || { [[ "$mode" == auto ]] && command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; }; then index=https://download.pytorch.org/whl/cu121; fi
"$python" -m pip install torch==2.4.1 --index-url "$index"
"$python" -m pip install -r scripts/requirements.txt
"$python" -m pip install -e '.[physics]'
"$python" -m pip check
"$python" scripts/prepare_examples.py
echo 'Ready. Put mind_diffusion.pkl and mind_autoencoder.pt in checkpoints/. See README.md.'
