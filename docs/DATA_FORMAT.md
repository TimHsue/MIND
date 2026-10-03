# Data format

`setup.sh` extracts `assets/examples/cell_001.npz` and `cell_002.npz` into this layout. It preserves an existing nonempty `data/` directory.

```text
data/
  holoplane/cell_001.npy
  holoplane/cell_002.npy
  voxel/cell_001.npy
  voxel/cell_002.npy
  points/cell_001.npy
  points/cell_002.npy
  dataset.json
  train.txt
  val.txt
```

Use identical sample basenames across directories. Arrays must be numeric NumPy arrays without pickled objects, NaN or infinity. Split files contain one filename per line. The two example split files overlap intentionally for smoke checks; real training and validation splits must be disjoint.

## Diffusion

Holoplane arrays are float32 `[3,32,64,64]`, before the training scale factor. Training flattens the first two axes into 96 channels, crops the top-left 32×32 corner and multiplies by 4. Do not pre-scale these arrays. `dataset.json` stores conditions keyed by filename:

```json
{
  "components": ["C11", "C12", "C44"],
  "profile": "v3",
  "label_space": "physical",
  "labels": {
    "cell_001.npy": [0.01848556, 0.00024582, 0.00182818],
    "cell_002.npy": [0.01641835, 0.00038225, 0.00164250]
  }
}
```

Physical values use material E=1 and nu=0.35. For already normalized conditions use `label_space: normalized`; if `label_space` is omitted, conditions are interpreted as normalized. Profile v3 transforms physical values as `[1.2*C11,3*(C12+0.01),5*C44]` exactly once.

## Geometry examples

Each bundled NPZ contains `holoplane`, `voxel` (float32 128³ signed-distance grid, negative inside and positive outside), `points` (float32 `[4096,4]`, columns x,y,z,SDF with coordinates in [-1,1] and negative SDF inside) and `condition` (physical `[3]`). Points and voxel use a consistent xyz order. The encoder expects signed-distance voxel values, rather than a solid-mask array. Voxel/points are derived from the decoded 64³ field; conditions describe the cropped 50³ evaluation cell.

**Geometry extent and padding:** the full SDF grid covers `[-1,1]³`, while the microstructure cell occupies the central approximately 80% of each axis, corresponding to `[-0.8,0.8]³`. The outer approximately 10% on each side is padding. Here, 80% means linear extent, not solid volume fraction: pores remain inside the cell. Preserve this padding when preparing encoder inputs; do not crop the cell and stretch it to fill the full SDF grid.

The SDF output is `[R,R,R]`, axes xyz, sampled on [-1,1].

## Joint autoencoder training

Joint training requires a paired float32 displacement array `[18,64,64,64]` for each sample, with three displacement components for each of six strain loads. Provide `--dataset_voxel`, `--dataset_occup` (SDF points), `--dataset_elastic_tensor`, `--dataset_list`, `--dataset_list_vali`, `--dataset_pro`, `--dataset_pro_vali`, and `--log_dir`; set `--resolution 128 --channels 32 --aggregate_fn sum --batch_size 1`. Property JSON uses `labels` keyed by filename with physical conditions.

Unlike the diffusion split files above, AE split files list sample basenames without `.npy`; the joint loader appends that extension. Property JSON keys still include `.npy`. Use separate, disjoint AE train/validation splits. See the [complete training command](../README.md#autoencoder-training). For the full dataset and checkpoints, contact [TimHsue@gmail.com](mailto:TimHsue@gmail.com).

SDF point coordinates use xyz order. Displacement arrays use the following coordinate and load ordering: spatial axes yxz on [-0.8,0.8], loads `[xx,yy,zz,xy,yz,xz]`, with consecutive x/y/z components for each load.

```bash
bash scripts/train_holoplane.sh --help
```
