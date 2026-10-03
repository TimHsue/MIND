"""Extract the bundled examples without replacing an existing dataset."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def prepare_examples(data):
    data = Path(data)
    if data.exists() and any(data.iterdir()):
        print(f'Existing dataset preserved: {data}')
        return
    paths = sorted((ROOT / 'assets/examples').glob('cell_*.npz'))
    if len(paths) != 2:
        raise RuntimeError('Expected two bundled examples')
    labels = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as item:
            for key in ('holoplane', 'voxel', 'points'):
                folder = data / key
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder / (path.stem + '.npy'), item[key])
            labels[path.stem + '.npy'] = item['condition'].tolist()
    metadata = dict(components=['C11', 'C12', 'C44'], profile='v3', label_space='physical', labels=labels)
    (data / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    for split in ('train', 'val'):
        (data / (split + '.txt')).write_text('\n'.join(labels) + '\n', encoding='utf-8')
    print(f'Prepared two examples: {data}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    prepare_examples(args.data_root)


if __name__ == '__main__':
    main()
