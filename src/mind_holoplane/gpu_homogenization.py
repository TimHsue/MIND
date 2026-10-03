"""GPU periodic homogenization with double precision and Jacobi-preconditioned CG."""
from pathlib import Path
import time
import types

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parent / 'native/native_full.py'


def evaluate(phase, device='cuda'):
    phase = np.asarray(phase, dtype=bool)
    if phase.ndim != 3 or len(set(phase.shape)) != 1 or phase.shape[0] < 2 or not phase.any():
        raise ValueError('Expected a nonempty cubic solid grid, resolution >=2')
    source = SOURCE.read_text(encoding='utf-8')
    index_expression = 'self.__K_indices.repeat(1, 2)[:, self.__K_sortidx][:, mask]'
    if source.count('.float().to(device)') != 2 or source.count(index_expression) != 1:
        raise RuntimeError('The native assembly source does not match the GPU adapter')
    source = source.replace(index_expression, 'torch.cat([self.__K_indices, self.__K_indices.flip(0)], dim=1)[:, self.__K_sortidx][:, mask]')
    source = source.replace('.float().to(device)', '.double().to(device)')
    module = types.ModuleType('mind_native_full_double')
    exec(compile(source, str(SOURCE), 'exec'), module.__dict__)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    start = time.monotonic()
    try:
        n = phase.shape[0]
        cache = str(SOURCE.parent / 'native_fem_cache.npz')
        assembly = module.homogenization(cache, n, n, n, 1, 1, 1, torch.device(device))
        from mind_holoplane.homogenization import isotropic_C
        material = torch.from_numpy(isotropic_C(1, .35)).to(device)
        assembly.macro_deformation(material)
        rho = torch.tensor(1e-6 + (1 - 1e-6) * phase.reshape(-1), device=device, dtype=torch.float64)
        stiffness, force = assembly.assembly(rho, .35)
        indices = stiffness.indices()
        diagonal_mask = indices[0] == indices[1]
        diagonal = torch.zeros(stiffness.shape[0], device=device, dtype=torch.float64)
        diagonal.index_add_(0, indices[0, diagonal_mask], stiffness.values()[diagonal_mask])
        if not bool((diagonal > 0).all()):
            raise ValueError('The assembled stiffness has a nonpositive diagonal')
        rhs_norm = force.norm(dim=0)
        zero_load = rhs_norm < 1e-13
        solve_rhs = force.clone()
        solve_rhs[:, zero_load] = 0
        displacement = module.linear_cg(
            lambda rhs: torch.sparse.mm(stiffness, rhs), solve_rhs, tolerance=1e-10,
            max_iter=3000, eps=1e-30, preconditioner=lambda rhs: rhs / diagonal[:, None])
        displacement[:, zero_load] = 0
        residual = torch.sparse.mm(stiffness, displacement) - force
        raw_relative_residual = residual.norm(dim=0) / rhs_norm.clamp_min(1e-30)
        effective_residual = torch.where(zero_load, torch.zeros_like(raw_relative_residual), raw_relative_residual)
        cells = assembly._homogenization__cellseq
        dofs = (3 * cells[:, :, None] + torch.arange(3, device=device)).reshape(-1, 24)
        energy = torch.zeros(6, 6, device=device, dtype=torch.float64)
        for begin in range(0, len(rho), 4096):
            end = begin + 4096
            local = assembly.U0[None] - displacement[dofs[begin:end]]
            energy += torch.einsum('eia,ij,ejb,e->ab', local, assembly.K0, local, rho[begin:end])
        force_work = (assembly.U0.T @ assembly.K0 @ assembly.U0) * rho.sum() - force.T @ displacement
        symmetry = float((force_work - force_work.T).norm() / force_work.norm().clamp_min(1e-30))
        energy_error = float((energy - force_work).norm() / force_work.norm().clamp_min(1e-30))
        eigenvalues = torch.linalg.eigvalsh((force_work + force_work.T) / 2)
        quality = bool(effective_residual.max() <= 1e-8 and symmetry <= 1e-6
                       and energy_error <= 1e-6 and eigenvalues.min() >= -1e-10)
        order = [0, 1, 2, 4, 5, 3]
        matrix = energy[order][:, order].detach().cpu().numpy()
        return {
            'status': 'completed' if quality else 'solver_quality_failure',
            'quality_passed': quality, 'C': matrix.tolist(), 'C_energy': matrix.tolist(),
            'C_force_work': force_work[order][:, order].detach().cpu().tolist(),
            'predicted': [float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[3, 3])],
            'native_load_order': ['xx', 'yy', 'zz', 'xy', 'yz', 'xz'],
            'reported_load_order': ['xx', 'yy', 'zz', 'yz', 'xz', 'xy'],
            'residual_per_load': effective_residual[order].cpu().tolist(),
            'native_residual_per_load': effective_residual.cpu().tolist(),
            'raw_relative_residual_per_load': raw_relative_residual[order].cpu().tolist(),
            'zero_RHS_mask': zero_load[order].cpu().tolist(),
            'zero_RHS_policy': 'For RHS norm below 1e-13, use a zero solution and report zero effective relative residual',
            'RHS_norm_per_load': rhs_norm[order].cpu().tolist(),
            'absolute_residual_per_load': residual.norm(dim=0)[order].cpu().tolist(),
            'symmetry_relative_error': symmetry, 'energy_identity_relative_error': energy_error,
            'eigenvalues': eigenvalues.cpu().tolist(), 'seconds': time.monotonic() - start,
            'backend': 'gpu_cg_jacobi_float64',
            'physics': {
                'E': 1, 'nu': .35, 'void_ratio': 1e-6, 'cell': [1, 1, 1], 'periodic': True,
                'Voigt': ['xx', 'yy', 'zz', 'yz', 'xz', 'xy'], 'engineering_shear': True,
                'linear_cg_denominator_floor': 1e-30, 'preconditioner': 'Jacobi',
                'cg_tolerance': 1e-10, 'maxiter': 3000,
                'residual_tolerance': 1e-8, 'tensor_tolerance': 1e-6,
            },
        }
    finally:
        torch.set_default_dtype(previous_dtype)
