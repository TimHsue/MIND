"""Check periodic Hex8 against homogeneous elasticity and an exact laminate."""
import argparse
import json
from pathlib import Path

import numpy as np

from mind_holoplane.homogenization import homogenize, isotropic_C


def laminate_tensor():
    # x-normal laminate: xx, xz, xy tractions are constant through its layers.
    a, b = [0, 4, 5], [1, 2, 3]
    phases = [isotropic_C(), 0.1 * isotropic_C()]
    H, Q, R = np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    for C in phases:
        inv = np.linalg.inv(C[np.ix_(a, a)])
        coupling = C[np.ix_(a, b)]
        H += 0.5 * inv
        Q += 0.5 * inv @ coupling
        R += 0.5 * (C[np.ix_(b, b)] - coupling.T @ inv @ coupling)
    A = np.linalg.inv(H)
    effective = np.empty((6, 6))
    effective[np.ix_(a, a)] = A
    effective[np.ix_(a, b)] = A @ Q
    effective[np.ix_(b, a)] = Q.T @ A
    effective[np.ix_(b, b)] = R + Q.T @ A @ Q
    return effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('outputs/homogenization_controls.json'))
    parser.add_argument('--load-workers', type=int, default=1)
    args = parser.parse_args()
    np.random.seed(0)
    rows = []
    for name in ['solid', 'laminate']:
        phase = np.ones((8, 8, 8), dtype=bool)
        ratio, expected = 1e-6, isotropic_C()
        if name == 'laminate':
            phase[4:] = False
            ratio, expected = 0.1, laminate_tensor()
        result = homogenize(phase, void_ratio=ratio, load_workers=args.load_workers)
        error = float(np.linalg.norm(np.array(result['C']) - expected) / np.linalg.norm(expected))
        if error > 1e-7:
            raise AssertionError(f'{name}: analytic relative error {error}')
        result.update(control=name, expected_C=expected.tolist(), analytic_relative_error=error)
        rows.append(result)
    try:
        homogenize(np.zeros((4, 4, 4), dtype=bool))
    except ValueError:
        empty_rejected = True
    else:
        raise AssertionError('An empty solid must not count as a physical homogenization')
    result = dict(status='passed', controls=rows, empty_geometry_rejected=empty_rejected,
                  limit='Solver controls only; generated structures require their own physical checks')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'analytic_errors': [x['analytic_relative_error'] for x in rows]}))


if __name__ == '__main__':
    main()
