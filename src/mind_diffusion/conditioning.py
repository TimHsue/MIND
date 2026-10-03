"""Normalize and denormalize physical C11/C12/C44 conditions."""
import numpy as np

COMPONENTS = ['C11', 'C12', 'C44']


def normalize_C(values, profile):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError('Expected finite [C11,C12,C44] values')
    if profile != 'v3':
        raise ValueError('Unknown condition profile')
    scale = np.array([1.2, 3.0, 5.0])
    return (array + np.array([0.0, 0.01, 0.0])) * scale


def denormalize_C(values, profile):
    if profile != 'v3':
        raise ValueError('Unknown condition profile')
    scale = np.array([1.2, 3.0, 5.0])
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError('Expected finite [C11,C12,C44] labels')
    return array / scale - np.array([0.0, 0.01, 0.0])
