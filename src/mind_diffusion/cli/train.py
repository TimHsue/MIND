"""Train MIND conditional diffusion on holoplane arrays."""
import json
import os
from pathlib import Path

import click
import torch

from mind_diffusion import dnnlib
from mind_diffusion.torch_utils import distributed as dist
from mind_diffusion.training import training_loop


@click.command()
@click.option('--outdir', type=click.Path(file_okay=False), required=True)
@click.option('--data', type=click.Path(exists=True, file_okay=False), required=True)
@click.option('--name-list', type=click.Path(exists=True, dir_okay=False))
@click.option('--conditions', type=click.Path(exists=True, dir_okay=False), required=True)
@click.option('--batch', type=click.IntRange(min=1), default=64)
@click.option('--batch-gpu', type=click.IntRange(min=1))
@click.option('--total-kimg', type=click.FloatRange(min=0, min_open=True), default=20000)
@click.option('--lr', type=click.FloatRange(min=0, min_open=True), default=1e-4)
@click.option('--workers', type=click.IntRange(min=0), default=4)
@click.option('--device', type=click.Choice(['cpu', 'cuda']), default='cuda')
@click.option('--model-channels', type=click.IntRange(min=8), default=192)
@click.option('--channel-mult', default='2,2,3,4,2')
@click.option('--num-blocks', type=click.IntRange(min=1), default=4)
@click.option('--label-fourier-scale', default='10,10,10')
@click.option('--condition-multiplier', '--label-scale', 'label_scale', type=float, default=1.0)
@click.option('--condition-profile', type=click.Choice(['v3']), default='v3')
@click.option('--crop-resolution', type=click.IntRange(min=0), default=32)
@click.option('--label-dropout', type=click.FloatRange(min=0, max=1), default=0.1)
@click.option('--latent-mean', type=float, default=0.0)
@click.option('--latent-scale', type=click.FloatRange(min=0, min_open=True), default=4.0)
@click.option('--resume', type=click.Path(exists=True, dir_okay=False))
@click.option('--resume-state', type=click.Path(exists=True, dir_okay=False))
@click.option('--resume-kimg', type=click.FloatRange(min=0), default=None, help='Override restored progress; otherwise use the training-state counter.')
@click.option('--seed', type=int, default=42)
def main(outdir, data, name_list, conditions, batch, batch_gpu, total_kimg, lr,
         workers, device, model_channels, channel_mult, num_blocks, label_scale,
         label_dropout, latent_mean, latent_scale, resume, resume_state,
         resume_kimg, seed, label_fourier_scale, condition_profile, crop_resolution):
    if device == 'cuda' and not torch.cuda.is_available():
        raise click.ClickException('CUDA is unavailable; use --device cpu for a small smoke run.')
    try:
        multipliers = [int(x) for x in channel_mult.split(',')]
        if not multipliers or min(multipliers) < 1:
            raise ValueError()
    except ValueError:
        raise click.BadParameter('Use positive integers separated by commas.', param_hint='--channel-mult')
    dist.init(device=device)
    target = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if device == 'cuda' else torch.device('cpu')
    Path(outdir).mkdir(parents=True, exist_ok=True)
    dist.set_run_dir(outdir)
    dataset_kwargs = dnnlib.EasyDict(
        class_name='mind_diffusion.training.dataset.HoloplaneFolderDataset', path=str(data),
        name_list_file=name_list, conditions=conditions, use_labels=conditions is not None,
        label_scale=label_scale, latent_mean=latent_mean, latent_scale=latent_scale,
        crop_resolution=crop_resolution, condition_profile=condition_profile,
        xflip=False, cache=False,
    )
    try:
        scales = [float(x) for x in label_fourier_scale.split(',')]
    except ValueError:
        raise click.BadParameter('Expected three finite positive Fourier scales', param_hint='--label-fourier-scale')
    if len(scales) != 3 or not all(0 < value < float('inf') for value in scales):
        raise click.BadParameter('Expected three Fourier scales', param_hint='--label-fourier-scale')
    network_kwargs = dnnlib.EasyDict(
        class_name='mind_diffusion.training.networks.MINDDenoiser',
        model_channels=model_channels,
        channel_mult=multipliers, channel_mult_noise=1, resample_filter=[1, 1],
        num_blocks=num_blocks, label_dropout=label_dropout,
        label_scale=scales, dropout=0.0,
    )
    config = dict(dataset_kwargs=dataset_kwargs, network_kwargs=network_kwargs,
                  batch=batch, total_kimg=total_kimg, lr=lr, seed=seed, device=str(target),
                  resume=resume, resume_state=resume_state, resume_kimg=resume_kimg)
    if dist.get_rank() == 0:
        (Path(outdir) / 'training_options.json').write_text(json.dumps(config, indent=2) + '\n')
    training_loop.training_loop(
        run_dir=os.path.abspath(outdir), dataset_kwargs=dataset_kwargs,
        data_loader_kwargs=dict(num_workers=workers, pin_memory=device == 'cuda'),
        network_kwargs=network_kwargs,
        loss_kwargs=dict(class_name='mind_diffusion.training.loss.EDMLoss', sigma_data=0.5),
        optimizer_kwargs=dict(class_name='torch.optim.Adam', lr=lr, betas=[0.9, 0.999], eps=1e-8),
        seed=seed, batch_size=batch, batch_gpu=batch_gpu, total_kimg=total_kimg,
        resume_pkl=resume, resume_state_dump=resume_state, resume_kimg=resume_kimg,
        snapshot_ticks=1, state_dump_ticks=1, device=target,
    )


if __name__ == '__main__':
    main()
