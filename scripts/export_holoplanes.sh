#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PYTHON" -m mind_holoplane.cli.export "$@"
