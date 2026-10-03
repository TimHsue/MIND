#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PYTHON" scripts/train.py "$@"
