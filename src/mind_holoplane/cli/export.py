"""Export full holoplanes from voxel arrays using a property-conditioned MIND encoder."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mind_holoplane.training.holoplane_ae import HoloplaneEncoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--voxel-dir', type=Path, required=True)
    parser.add_argument('--name-list', type=Path, required=True)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--out-resolution', type=int, default=64)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = HoloplaneEncoder(args.resolution, out_channels=args.channels,
                             out_resolution=args.out_resolution, label_dim=3, label_scale=[10, 10, 10])
    model.load_state_dict(state['encoder_state_dict'], strict=True)
    model = model.to(args.device).eval()
    properties = json.loads(args.conditions.read_text())
    if properties.get('label_space', 'physical') != 'physical':
        parser.error('The encoder requires physical C11,C12,C44 conditions')
    properties = dict(properties.get('labels', properties))
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name in args.name_list.read_text().splitlines():
        name = name.strip()
        if not name:
            continue
        filename = name if name.endswith('.npy') else name + '.npy'
        array = np.load(args.voxel_dir / filename, allow_pickle=False).reshape((args.resolution,) * 3)
        voxel = torch.from_numpy(array.astype(np.float32))
        for axis in range(3):
            voxel = (voxel + voxel.flip(axis)) / 2
        labels = torch.tensor(properties[filename], dtype=torch.float32).reshape(1, 3).to(args.device)
        if not torch.isfinite(voxel).all() or not torch.isfinite(labels).all():
            raise ValueError(f'{filename}: non-finite voxel or labels')
        with torch.inference_mode():
            latent = model(voxel.unsqueeze(0).to(args.device), labels)[:, 0].cpu().numpy()
        if not np.isfinite(latent).all():
            raise FloatingPointError(f'{filename}: non-finite latent')
        np.save(args.outdir / filename, latent)
        print(filename, latent.shape)


if __name__ == '__main__':
    main()
