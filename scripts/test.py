"""Check bundled examples, condition normalization and CPU physics."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

from mind_diffusion.conditioning import normalize_C
from mind_diffusion.training.dataset import HoloplaneFolderDataset
from prepare_examples import prepare_examples

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-model', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='mind-examples-') as directory:
        data = Path(directory)
        prepare_examples(data)
        dataset = HoloplaneFolderDataset(path=str(data / 'holoplane'), conditions=str(data / 'dataset.json'),
            use_labels=True, condition_profile='v3', crop_resolution=32, latent_scale=4)
        try:
            for i, path in enumerate(sorted((ROOT / 'assets/examples').glob('cell_*.npz'))):
                with np.load(path, allow_pickle=False) as example:
                    image, label = dataset[i]
                    np.testing.assert_allclose(image, example['holoplane'].reshape(96, 64, 64)[:, :32, :32] * 4)
                    np.testing.assert_allclose(label, normalize_C(example['condition'], 'v3'), rtol=1e-6)
                    assert example['voxel'].shape == (128, 128, 128)
                    assert example['points'].shape == (4096, 4)
                    assert np.isfinite(example['voxel']).all()
                    assert example['voxel'].min() < 0 < example['voxel'].max()
                    assert np.isfinite(example['points']).all()
        finally:
            dataset.close()
        subprocess.run([sys.executable, str(ROOT / 'scripts/verify_homogenization.py')], cwd=ROOT, check=True)
        if args.with_model:
            from mind_diffusion.checkpoint_io import load_snapshot
            model = load_snapshot(ROOT / 'checkpoints/mind_diffusion.pkl')['ema']
            assert model.img_resolution == 32 and model.img_channels == 96
            del model
            subprocess.run([sys.executable, '-m', 'mind_holoplane.cli.decode',
                '--checkpoint', str(ROOT / 'checkpoints/mind_autoencoder.pt'),
                '--latent', str(data / 'holoplane/cell_001.npy'), '--outdir', str(ROOT / 'outputs/test_decode'),
                '--device', 'cpu', '--resolution', '16'], cwd=ROOT, check=True)
    print('Checks passed.' + (' Both checkpoints loaded; example decoded.' if args.with_model else ''))


if __name__ == '__main__':
    main()
