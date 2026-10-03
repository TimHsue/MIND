"""Compute the elastic tensor of a prepared periodic voxel cell."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solid', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    solid = np.load(args.solid, allow_pickle=False)
    if solid.ndim != 3 or len(set(solid.shape)) != 1 or solid.dtype != bool or solid.shape[0] < 2:
        parser.error('Expected a cubic boolean solid array with at least two cells per axis')
    if not solid.any():
        parser.error('The cell must contain solid material')
    if args.device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable; use --device cpu')
        from mind_holoplane.gpu_homogenization import evaluate
        result = evaluate(solid, 'cuda')
    else:
        from mind_holoplane.homogenization import homogenize
        result = homogenize(solid, young=1, poisson=.35, void_ratio=1e-6, rtol=1e-10, maxiter=3000, load_workers=1)
        result['predicted'] = [result['C11'], result['C12'], result['C44']]
        result['residual_per_load'] = [load['relative_residual'] for load in result['loads']]
    result['quality_passed'] = bool(np.isfinite(result['C']).all() and np.isfinite(result['residual_per_load']).all()
        and max(result['residual_per_load']) <= 1e-8
        and np.isfinite(result['eigenvalues']).all() and min(result['eigenvalues']) >= -1e-10
        and np.isfinite([result['symmetry_relative_error'], result['energy_identity_relative_error']]).all()
        and result['symmetry_relative_error'] <= 1e-6 and result['energy_identity_relative_error'] <= 1e-6)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if not result['quality_passed']:
        raise RuntimeError('Physics checks failed; inspect the saved diagnostics')
    print(json.dumps(dict(passed=True, out=str(args.out), max_residual=max(result['residual_per_load']))))


if __name__ == '__main__':
    main()
