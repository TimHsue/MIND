#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then echo "Run bash setup.sh first (or set PYTHON)." >&2; exit 1; fi
