"""Sample holoplanes with conditional Heun sampling."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from mind_diffusion.checkpoint_io import load_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, help='Defaults to the released checkpoint')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--labels', help='Normalized C11,C12,C44 values, separated by commas')
    parser.add_argument('--raw-C', help='Physical C11,C12,C44 for material E=1 and nu=0.35')
    parser.add_argument('--condition-profile', choices=['v3'], default='v3')
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--noise-device', choices=['cpu', 'model'], default='cpu', help='Device used to generate initial noise; use cpu for the default seeded outputs')
    parser.add_argument('--sigma-max', type=float, default=80.0)
    parser.add_argument('--cfg-scale', type=float, default=7.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', help='Optional comma-separated exact seed list; length must equal count')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--keep-corner', action='store_true', help='Save the native 32x32 corner without mirror reconstruction')
    args = parser.parse_args()
    if args.count < 1 or args.steps < 2 or args.batch_size < 1:
        parser.error('count and batch-size must be >=1 and steps must be >=2')
    if not math.isfinite(args.sigma_max) or args.sigma_max <= 0.002 or not math.isfinite(args.cfg_scale):
        parser.error('sigma-max must be finite and >0.002; cfg-scale must be finite')
    try:
        seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else list(range(args.seed,args.seed+args.count))
    except ValueError:
        parser.error('seeds must be integers separated by commas')
    if any(not -(2**63) <= seed < 2**64 for seed in seeds):
        parser.error('seeds must fit the PyTorch seed range [-2**63, 2**64-1]')
    if len(seeds) != args.count:
        parser.error('seeds list length must equal count')
    if args.network is None:
        root = Path(__file__).resolve().parents[3]
        config_file = root / 'scripts/default.json'
        args.network = (root / json.loads(config_file.read_text())['dm']).resolve() if config_file.exists() else Path('checkpoints/mind_diffusion.pkl')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = load_snapshot(args.network)
    model = snapshot['ema'].to(args.device).eval()
    options = dict(snapshot.get('dataset_kwargs', {}))
    sidecar = args.network.with_name(args.network.name + '.metadata.json')
    if sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        options.update(metadata.get('runtime_dataset_kwargs', {}))
    checkpoint_profile = options.get('condition_profile')
    if args.condition_profile and checkpoint_profile and args.condition_profile != checkpoint_profile:
        parser.error(f'Checkpoint expects profile {checkpoint_profile}; do not substitute {args.condition_profile}')
    labels = None
    if model.label_dim:
        if bool(args.labels) == bool(args.raw_C):
            parser.error('Use exactly one of --labels (normalized) or --raw-C')
        vector = [float(x) for x in (args.labels or args.raw_C).split(',')]
        if args.raw_C:
            profile = args.condition_profile or options.get('condition_profile')
            if profile != 'v3':
                parser.error('Raw conditions require an explicit --condition-profile v3')
            from mind_diffusion.conditioning import normalize_C
            vector = normalize_C(vector, profile).tolist()
        labels = torch.tensor(vector, device=args.device)
        if labels.numel() != model.label_dim or not torch.isfinite(labels).all():
            parser.error(f'Expected {model.label_dim} finite condition values')
        labels = labels.reshape(1, -1) * options.get('label_scale', 1.0)
    elif args.labels or args.raw_C:
        parser.error('This network is unconditional and does not accept labels')
    args.outdir.mkdir(parents=True, exist_ok=True)
    for start in range(0, args.count, args.batch_size):
        size = min(args.batch_size, args.count-start)
        noise_device = 'cpu' if args.noise_device == 'cpu' else args.device
        noise = torch.stack([torch.randn((model.img_channels, model.img_resolution, model.img_resolution),
            generator=torch.Generator(device=noise_device).manual_seed(seeds[index]), device=noise_device)
            for index in range(start,start+size)]).to(args.device)
        batch_labels = labels.expand(size,-1) if labels is not None else None
        with torch.inference_mode():
            from mind_diffusion.cli.sampler import heun_sampler
            sample = heun_sampler(model, noise, batch_labels, num_steps=args.steps,
                sigma_max=args.sigma_max, cond_strength=args.cfg_scale,
                drop_labels=torch.ones_like(batch_labels).unsqueeze(-1))
        sample = sample / options.get('latent_scale', 1.0) + options.get('latent_mean', 0.0)
        for offset in range(size):
            array = sample[offset].float().cpu().numpy()
            if not np.isfinite(array).all():
                raise FloatingPointError('Sampler produced non-finite values')
            if array.shape[0] % 3 == 0:
                array = array.reshape(3, -1, model.img_resolution, model.img_resolution)
            if options.get('crop_resolution') and not args.keep_corner:
                array = np.concatenate([array, np.flip(array, axis=-1)], axis=-1)
                array = np.concatenate([array, np.flip(array, axis=-2)], axis=-2)
            np.save(args.outdir / f'sample_{start+offset:04d}.npy', array)
        del sample, noise
    result = dict(network=args.network.name, count=args.count, steps=args.steps, seed=seeds[0], seeds=seeds,
                  device=args.device, finite=True, shape=list(array.shape),
                  latent_mean=options.get('latent_mean', 0.0), latent_scale=options.get('latent_scale', 1.0))
    result['condition_profile'] = args.condition_profile or options.get('condition_profile')
    result['normalized_condition'] = vector if labels is not None else None
    result['corner_mirrored'] = bool(options.get('crop_resolution') and not args.keep_corner)
    result['cfg_scale'] = args.cfg_scale
    result['sampler'] = 'heun-cfg'
    result['sigma_max'] = args.sigma_max
    result['batch_size'] = args.batch_size
    result['noise_device'] = args.noise_device
    (args.outdir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
