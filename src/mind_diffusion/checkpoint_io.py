"""Load persistent checkpoints on a selected device."""
import io
import pickle
import sys

import torch

from mind_diffusion import dnnlib, torch_utils
from mind_diffusion.torch_utils import misc, persistence, training_stats


def load_snapshot(path, device='cpu'):
    for name, module in [('dnnlib', dnnlib), ('torch_utils', torch_utils),
                         ('torch_utils.persistence', persistence), ('torch_utils.misc', misc),
                         ('torch_utils.training_stats', training_stats)]:
        sys.modules.setdefault(name, module)

    class DeviceUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda data: torch.load(io.BytesIO(data), map_location=device, weights_only=False)
            return super().find_class(module, name)

    with open(path, 'rb') as stream:
        snapshot = DeviceUnpickler(stream).load()
    return snapshot
