#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: bash scripts/demo.sh [holoplane|diffusion] [options]'
  echo 'holoplane: existing holoplane -> OBJ (autoencoder weight only)'
  echo 'diffusion: physical condition -> holoplane -> OBJ (both weights)'
  echo 'Use a mode followed by --help for its options. Default mode: diffusion.'
  exit 0
fi
mode=diffusion
if [[ "${1:-}" == holoplane || "${1:-}" == diffusion ]]; then
  mode="$1"
  shift
fi
case "$mode" in
  diffusion) exec "$PYTHON" scripts/run_default.py "$@" ;;
  holoplane) exec "$PYTHON" scripts/run_geometry_demo.py "$@" ;;
esac
