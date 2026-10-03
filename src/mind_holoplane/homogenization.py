"""Six-load periodic, fully integrated Hex8 linear elastic homogenization.

Voigt order: xx, yy, zz, yz, xz, xy; engineering shear strains.
Unit cell edges have length one. The density scales Young's modulus,
including an explicit small void stiffness for SPD stabilization.
"""
import itertools
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg


def isotropic_C(young=1.0, poisson=0.35):
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    C = np.zeros((6, 6))
    C[:3, :3] = lam
    C[np.arange(3), np.arange(3)] += 2 * mu
    C[3:, 3:] = np.eye(3) * mu
    return C


def hex8(young, poisson, h):
    offsets = np.array(list(itertools.product([0, 1], repeat=3)))
    signs = 2 * offsets - 1
    C = isotropic_C(young, poisson)
    K = np.zeros((24, 24))
    for point in itertools.product([-1 / np.sqrt(3), 1 / np.sqrt(3)], repeat=3):
        factors = 1 + signs * np.asarray(point)
        grad = np.empty((8, 3))
        for axis in range(3):
            other = [j for j in range(3) if j != axis]
            grad[:, axis] = signs[:, axis] * factors[:, other].prod(1) / (4 * h)
        B = np.zeros((6, 24))
        for node, (x, y, z) in enumerate(grad):
            j = 3 * node
            B[0, j], B[1, j + 1], B[2, j + 2] = x, y, z
            B[3, j + 1], B[3, j + 2] = z, y
            B[4, j], B[4, j + 2] = z, x
            B[5, j], B[5, j + 1] = y, x
        K += B.T @ C @ B * (h / 2) ** 3
    strains = np.zeros((6, 3, 3))
    for i in range(3):
        strains[i, i, i] = 1
    for i, (a, b) in enumerate([(1, 2), (0, 2), (0, 1)], 3):
        strains[i, a, b] = strains[i, b, a] = 0.5
    affine = np.einsum('aij,nj->nia', strains, offsets * h).reshape(24, 6)
    np.testing.assert_allclose(affine.T @ K @ affine, C * h ** 3, atol=1e-14)
    return K, affine, offsets


def homogenize(phase, young=1.0, poisson=0.35, void_ratio=1e-6,
               rtol=1e-8, maxiter=3000, callback=None, load_workers=1):
    import pyamg
    start = time.perf_counter()
    phase = np.asarray(phase, dtype=bool)
    n = phase.shape[0]
    if phase.shape != (n, n, n) or n < 2:
        raise ValueError('Expected a cubic voxel grid, resolution >=2')
    if not 0 < void_ratio <= 1 or not -1 < poisson < 0.5:
        raise ValueError('Invalid material parameters')
    if not phase.any():
        raise ValueError('An empty geometry has no physical solid to homogenize')
    h = 1 / n
    element_K, affine, offsets = hex8(young, poisson, h)
    rho = np.where(phase, 1.0, void_ratio).ravel()
    cells = np.indices(phase.shape).reshape(3, -1).T
    nodes = (cells[:, None, :] + offsets[None]) % n
    ids = ((nodes[..., 0] * n + nodes[..., 1]) * n + nodes[..., 2]).astype(np.int32)
    dofs = (3 * ids[..., None] + np.arange(3, dtype=np.int32)).reshape(-1, 24)
    del cells, nodes, ids
    count = 3 * n ** 3
    shape = (len(rho), 24, 24)
    row = np.broadcast_to(dofs[:, :, None], shape).ravel()
    col = np.broadcast_to(dofs[:, None, :], shape).ravel()
    values = (rho[:, None, None] * element_K).ravel()
    K = coo_matrix((values, (row, col)), shape=(count, count)).tocsr()
    K.eliminate_zeros()
    del row, col, values
    force = np.zeros((count, 6))
    element_force = element_K @ affine
    for component in range(6):
        np.add.at(force[:, component], dofs.ravel(), (rho[:, None] * element_force[:, component]).ravel())
    # One periodic node fixes the three translation modes. No macroscopic strain is pinned.
    A = K[3:, 3:].tobsr(blocksize=(3, 3))
    B = np.tile(np.eye(3), (n ** 3 - 1, 1))
    if callback:
        callback(dict(stage='assembled', resolution=n, dofs=count - 3, nnz=A.nnz,
                      seconds=time.perf_counter() - start))
    multilevel = pyamg.smoothed_aggregation_solver(
        A, B=B, symmetry='symmetric', max_coarse=100,
        presmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
        postsmoother=('block_gauss_seidel', {'sweep': 'symmetric'}),
    )
    M = multilevel.aspreconditioner(cycle='V')
    u = np.zeros((count, 6))
    loads = [None] * 6
    def solve_load(component, progress):
        rhs = -force[3:, component]
        norm = np.linalg.norm(rhs)
        iterations = [0]
        def step(_):
            iterations[0] += 1
            if iterations[0] % 100 == 0:
                progress(dict(stage='cg_progress', resolution=n, component=component,
                              iterations=iterations[0], seconds=time.perf_counter() - start))
        if norm < 1e-13:
            info = 0
            residual = 0.0
            solution = np.zeros(count - 3)
        else:
            solution, info = cg(A, rhs, M=M, rtol=rtol, atol=0, maxiter=maxiter, callback=step)
            residual = float(np.linalg.norm(A @ solution - rhs) / norm)
        record = dict(component=component, cg_info=int(info), iterations=iterations[0],
                      relative_residual=residual, rhs_norm=float(norm))
        return solution, record

    def accept(solution, record):
        component = record['component']
        u[3:, component] = solution
        loads[component] = record
        if callback:
            callback(dict(stage='load_solved', resolution=n, **record,
                          seconds=time.perf_counter() - start))
        if (record['cg_info'] or not np.isfinite(record['relative_residual'])
                or record['relative_residual'] > rtol):
            raise RuntimeError(f'Unconverged homogenization load: {record}')
    # The six independent RHS share read-only matrix/preconditioner pages on Linux.
    # Solve strain loads sequentially by default.
    if load_workers == 1:
        for component in range(6):
            solution, record = solve_load(component, callback or (lambda _: None))
            accept(solution, record)
    else:
        import multiprocessing as mp
        if not 1 < load_workers <= 6 or 'fork' not in mp.get_all_start_methods():
            raise ValueError('Parallel loads require Linux fork and 2..6 workers')
        context = mp.get_context('fork')
        queue = context.Queue()
        def worker(component):
            try:
                solution, record = solve_load(component, lambda p: queue.put(('progress', p)))
                queue.put(('solved', (solution, record)))
            except Exception as error:
                queue.put(('error', repr(error)))
        active, next_load, completed = {}, 0, 0
        try:
            while completed < 6:
                while len(active) < load_workers and next_load < 6:
                    process = context.Process(target=worker, args=(next_load,))
                    process.start()
                    active[next_load] = process
                    next_load += 1
                kind, value = queue.get(timeout=3600)
                if kind == 'progress':
                    if callback:
                        callback(value)
                elif kind == 'error':
                    raise RuntimeError(value)
                else:
                    solution, record = value
                    active.pop(record['component']).join()
                    accept(solution, record)
                    completed += 1
        finally:
            for process in active.values():
                process.terminate()
                process.join()
            queue.close()
    C = (affine.T @ element_K @ affine) * rho.sum() + force.T @ u
    symmetry = float(np.linalg.norm(C - C.T) / max(np.linalg.norm(C), 1e-30))
    energy_C = np.zeros((6, 6))
    for begin in range(0, len(rho), 4096):
        end = begin + 4096
        local = affine[None] + u[dofs[begin:end]]
        gradient = np.einsum('ij,eja->eia', element_K, local, optimize=True)
        energy_C += np.einsum('eia,eib,e->ab', local, gradient, rho[begin:end], optimize=True)
    energy_error = float(np.linalg.norm(energy_C - C) / max(np.linalg.norm(C), 1e-30))
    eigenvalues = np.linalg.eigvalsh((C + C.T) / 2)
    if symmetry > 1e-6 or energy_error > 1e-6 or eigenvalues[0] < -1e-10:
        raise RuntimeError(f'Invalid tensor: symmetry={symmetry}, energy={energy_error}, eig={eigenvalues}')
    return dict(C=C.tolist(), C11=float(C[0, 0]), C12=float(C[0, 1]), C44=float(C[3, 3]),
                resolution=n, volume_fraction=float(phase.mean()), young=young, poisson=poisson,
                void_young_ratio=void_ratio, loads=loads, all_loads_converged=True,
                symmetry_relative_error=symmetry, energy_identity_relative_error=energy_error,
                eigenvalues=eigenvalues.tolist(), seconds=time.perf_counter() - start,
                solver='periodic Hex8, 2x2x2 Gauss, AMG-preconditioned CG',
                load_workers=load_workers,
                voigt_order=['xx', 'yy', 'zz', 'yz', 'xz', 'xy'])
