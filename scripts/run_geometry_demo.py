"""Decode an example or caller-supplied holoplane into SDF and OBJ."""
import argparse
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latent', type=Path, default=ROOT/'data/holoplane/cell_001.npy')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/mind_autoencoder.pt')
    parser.add_argument('--outdir', type=Path)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--chunk-size', type=int, default=8192)
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.checkpoint.is_file():
        parser.error('Place mind_autoencoder.pt in checkpoints/, or supply --checkpoint.')
    out = (args.outdir or ROOT/'outputs/holoplane_demo').resolve()
    latent = args.latent.resolve()
    if not latent.is_file():
        parser.error('Holoplane file missing. Run setup.sh or supply --latent.')
    subprocess.run([sys.executable, '-m', 'mind_holoplane.cli.decode',
                    '--checkpoint', str(args.checkpoint.resolve()), '--latent', str(latent),
                    '--outdir', str(out), '--resolution', str(args.resolution),
                    '--chunk-size', str(args.chunk_size), '--device', device], cwd=ROOT, check=True)
    from mind_holoplane.preview import save_geometry_preview
    save_geometry_preview(latent, out/'field.npy', out/'preview.png', 'Holoplane → SDF → OBJ')
    print(f'OBJ saved to {out / "mesh.obj"}')


if __name__ == '__main__':
    main()
