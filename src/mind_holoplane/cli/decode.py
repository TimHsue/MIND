"""Decode saved holoplanes with a MIND decoder checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from skimage.measure import marching_cubes

from mind_holoplane.training.holoplane_ae import HoloplaneDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Defaults to the released autoencoder')
    parser.add_argument('--latent', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=8192)
    parser.add_argument('--aggregate-fn', choices=['sum'], default='sum')
    parser.add_argument('--sym', type=int, choices=[1, 8, 48], default=48)
    parser.add_argument('--level', type=float, default=0.0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.checkpoint is None:
        root = Path(__file__).resolve().parents[3]
        config_file = root / 'scripts/default.json'
        args.checkpoint = (root / json.loads(config_file.read_text())['ae']).resolve() if config_file.exists() else Path('checkpoints/mind_autoencoder.pt')
    if args.resolution < 2 or args.chunk_size < 1:
        parser.error('resolution must be >=2 and chunk-size must be >=1')
    array = np.load(args.latent, allow_pickle=False)
    if array.ndim == 3 and array.shape[0] % 3 == 0:
        array = array.reshape(3, -1, array.shape[-2], array.shape[-1])
    if array.ndim != 4 or array.shape[0] != 3 or not np.isfinite(array).all():
        raise ValueError(f'Expected finite [3,C,H,W] latent, got {array.shape}')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = HoloplaneDecoder(array.shape[1], aggregate_fn=args.aggregate_fn,
                             use_tanh=bool(state.get('decoder_kwargs', {}).get('use_tanh', False)))
    model.load_state_dict(state['decoder_state_dict'], strict=True)
    model = model.to(args.device).eval()
    latent = torch.from_numpy(array.astype(np.float32)).unsqueeze(1).to(args.device)
    axis = torch.linspace(-1, 1, args.resolution, device=args.device)
    coordinates = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    values = []
    with torch.inference_mode():
        for chunk in coordinates.split(args.chunk_size):
            values.append(model(latent, chunk.unsqueeze(0).to(args.device), sym=args.sym).flatten().cpu())
    field = torch.cat(values).numpy().reshape((args.resolution,) * 3)
    if not np.isfinite(field).all():
        raise FloatingPointError('Decoder produced non-finite values')
    if not float(field.min()) < args.level < float(field.max()):
        raise ValueError(f'No surface at level {args.level}; field range is {field.min()} .. {field.max()}')
    vertices, faces, _, _ = marching_cubes(field, level=args.level, spacing=(2 / (args.resolution - 1),) * 3)
    vertices -= 1
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / 'field.npy', field)
    with (args.outdir / 'mesh.obj').open('w') as stream:
        for x, y, z in vertices:
            stream.write(f'v {x:.8f} {y:.8f} {z:.8f}\n')
        for a, b, c in faces + 1:
            stream.write(f'f {a} {b} {c}\n')
    result = dict(checkpoint=args.checkpoint.name, latent=args.latent.name,
                  strict_load=True, finite=True, field_min=float(field.min()), field_max=float(field.max()),
                  resolution=args.resolution, sym=args.sym, level=args.level,
                  vertices=len(vertices), faces=len(faces), device=args.device)
    (args.outdir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
