#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
exec "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  -m mind_diffusion.cli.train "$@"
