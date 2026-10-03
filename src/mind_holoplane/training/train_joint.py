"""Train the MIND joint holoplane autoencoder and field decoders."""

import os
import argparse
import random
import numpy as np
import torch
from .dataset import VoxelOccupancyDataset
from .holoplane_ae import HoloplaneDecoder, HoloplaneEncoder, ElasticTensorDecoder
from torch.utils.data import DataLoader, Subset
import time
from tensorboardX import SummaryWriter
from .utils.regularization import edr_loss, l2_reg, tv_reg, sym_loss_fun
from tqdm import tqdm

from .vis_tools import visualize_holoplane, visualize_occupancy, visualize_occupancy_error, visualize_sdf_clip


def existsOrMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def load_state_dict(model, state_dict):
    model.load_state_dict({key.removeprefix('module.'): value for key, value in state_dict.items()}, strict=True)


def main():
    parser = argparse.ArgumentParser(description='Train the MIND joint holoplane autoencoder.')
    parser.add_argument('--dataset_voxel', type=str,
                    help='directory to voxel SDF arrays', required=True)
    parser.add_argument('--dataset_occup', type=str,
                    help='directory to xyz/SDF point arrays', required=True)
    parser.add_argument('--dataset_elastic_tensor', type=str,
                    help='directory to 18-channel displacement references', required=True)
    parser.add_argument('--dataset_list', type=str,
                    help='train filelist', required=True)
    parser.add_argument('--dataset_list_vali', type=str,
                    help='vali filelist', required=True)
    parser.add_argument('--dataset_pro', type=str,
                    help='JSON physical training conditions', required=True)
    parser.add_argument('--dataset_pro_vali', type=str,
                    help='JSON physical validation conditions', required=True)
    parser.add_argument('--log_dir', type=str,
                    help='directory to log', required=True)

    parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='number of scenes per batch')
    parser.add_argument('--points_batch_size', type=int, default=8192, required=False,
                    help='SDF query points sampled per case')
    parser.add_argument('--log_every', type=int, default=1, required=False)
    parser.add_argument('--val_every', type=int, default=1, required=False)
    parser.add_argument('--vis_every', type=int, default=1, required=False)
    parser.add_argument('--save_every', type=int, default=4, required=False)

    parser.add_argument('--load_ckpt_path', type=str, default=None, required=False,
                    help='checkpoint to continue training from')
    parser.add_argument('--checkpoint_path', type=str, default='outputs/ae_training/checkpoints', required=False,
                    help='where to save model checkpoints')

    parser.add_argument('--resolution', type=int, default=128, required=False,
                    help='input voxel resolution')
    parser.add_argument('--channels', type=int, default=32, required=False,
                    help='holoplane depth')
    parser.add_argument('--aggregate_fn', choices=['sum'], default='sum', required=False,
                    help='function for aggregating holoplane features')

    parser.add_argument('--steps_per_batch', type=int, default=1, required=False,
                    help='Gradient accumulation passes per loaded batch')
    parser.add_argument('--edr_val', type=float, default=None, required=False,
                    help='If specified, use explicit density regularization with the specified offset distance value.')
    parser.add_argument('--use_tanh', default=False, required=False, action='store_true',
                    help='Whether to use tanh to clamp holoplanes to [-1, 1].')

    parser.add_argument('--e_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--p_lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--vis_chunk_size', type=int, default=8192)

    args = parser.parse_args()
    if min(args.steps_per_batch, args.batch_size, args.points_batch_size, args.epochs, args.val_every, args.save_every, args.log_every, args.vis_chunk_size) < 1 or args.workers < 0 or args.vis_every < 0:
        parser.error('Counts and intervals must be positive; workers and vis_every may be zero')

    if args.resolution < 64 or args.resolution & (args.resolution - 1) or args.channels < 1:
        parser.error('Voxel resolution must be a power of two >=64; channels must be positive')
    if not np.isfinite([args.e_lr, args.d_lr, args.p_lr, args.grad_clip]).all() or min(args.e_lr, args.d_lr, args.p_lr) <= 0:
        parser.error('Learning rates must be finite and positive; grad_clip must be finite')
    if not torch.cuda.is_available():
        parser.error('Joint AE training requires CUDA')
    # The shell launcher supplies the distributed process environment.
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    # initialize the process group
    print("preparing process group...")
    torch.distributed.init_process_group("nccl", rank=int(os.environ['RANK']), world_size=world_size)

    setRandomSeed(args.seed + local_rank)
    # create the model and move it to the local GPU
    device = torch.device("cuda", local_rank)
    print("device locked ", local_rank)
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    torch.set_num_threads(16)


    ckpt_path = None
    if local_rank == 0:
        tb_path = args.log_dir
        existsOrMkdir(tb_path)
        writer = SummaryWriter(tb_path, "base")
        if args.checkpoint_path:
            ckpt_path = args.checkpoint_path
            os.makedirs(ckpt_path, exist_ok=True)

    checkpoint = None
    if args.load_ckpt_path:
        checkpoint = torch.load(args.load_ckpt_path, map_location="cpu", weights_only=False)
        if 'consm_decoder_state_dict' in checkpoint:
            checkpoint['elastic_tensor_decoder_state_dict'] = checkpoint['consm_decoder_state_dict']
        args.use_tanh = args.use_tanh or checkpoint.get('decoder_kwargs', {}).get('use_tanh', False)
        print(f'Loaded AE checkpoint: {args.load_ckpt_path}')

    dataset = VoxelOccupancyDataset(
        resolution=args.resolution,
        file_list_name=args.dataset_list,
        voxel_dir=args.dataset_voxel,
        occup_dir=args.dataset_occup,
        elastic_tensor_dir=args.dataset_elastic_tensor,
        dataset_pro=args.dataset_pro,
        points_batch_size=args.points_batch_size,
        return_name=False,
        return_property=True,
        return_elastic_tensor=True,
        device=device)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, sampler=sampler)

    dataset_vali = VoxelOccupancyDataset(
        resolution=args.resolution,
        file_list_name=args.dataset_list_vali,
        voxel_dir=args.dataset_voxel,
        occup_dir=args.dataset_occup,
        elastic_tensor_dir=args.dataset_elastic_tensor,
        dataset_pro=args.dataset_pro_vali,
        points_batch_size=args.points_batch_size,
        return_name=True,
        return_elastic_tensor=True,
        return_property=True,
        device=device)
    validation_indices = range(int(os.environ['RANK']), len(dataset_vali), world_size)
    dataloader_vali = DataLoader(Subset(dataset_vali, validation_indices), batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers)

    auto_encoder = HoloplaneEncoder(
        resolution=args.resolution,
        in_channels=1,
        out_channels=args.channels,
        label_dim=3,
        label_scale=[10, 10, 10],
        device=device).to(device)

    auto_decoder = HoloplaneDecoder(
        channels=args.channels,
        aggregate_fn=args.aggregate_fn,
        use_coord=False,
        use_tanh=args.use_tanh).to(device)

    elastic_tensor_decoder = ElasticTensorDecoder(
        resolution=64,
        in_channels=args.channels,
        mid_channels=18,
        out_channels=3,
        aggregate_fn=args.aggregate_fn).to(device)


    if checkpoint:
        if 'encoder_state_dict' in checkpoint:
            load_state_dict(auto_encoder, checkpoint['encoder_state_dict'])
        if 'decoder_state_dict' in checkpoint:
            load_state_dict(auto_decoder, checkpoint['decoder_state_dict'])
        if 'elastic_tensor_decoder_state_dict' in checkpoint:
            load_state_dict(elastic_tensor_decoder, checkpoint['elastic_tensor_decoder_state_dict'])


    auto_encoder = torch.nn.parallel.DistributedDataParallel(auto_encoder, device_ids=[local_rank], output_device=local_rank)
    auto_decoder = torch.nn.parallel.DistributedDataParallel(auto_decoder, device_ids=[local_rank], output_device=local_rank)
    elastic_tensor_decoder = torch.nn.parallel.DistributedDataParallel(elastic_tensor_decoder, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam([
        {'params': auto_encoder.parameters(), 'lr': args.e_lr},
        {'params': auto_decoder.parameters(), 'lr': args.d_lr},
        {'params': elastic_tensor_decoder.parameters(), 'lr': args.p_lr},
    ], betas=(0.9, 0.999))
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for group, lr in zip(optimizer.param_groups, (args.e_lr, args.d_lr, args.p_lr)):
            group['lr'] = lr

    elastic_tensor_x = np.linspace(-0.8, 0.8, 64)
    elastic_tensor_y = np.linspace(-0.8, 0.8, 64)
    elastic_tensor_z = np.linspace(-0.8, 0.8, 64)
    elastic_tensor_X, elastic_tensor_Y, elastic_tensor_Z = np.meshgrid(elastic_tensor_x, elastic_tensor_y, elastic_tensor_z)
    elastic_tensor_voxel = np.stack([elastic_tensor_X.ravel(), elastic_tensor_Y.ravel(), elastic_tensor_Z.ravel()], axis=-1)
    elastic_tensor_voxel = torch.Tensor(elastic_tensor_voxel).unsqueeze(0).to(device) # 1, 64 * 64 * 64, 3

    auto_encoder.train()
    auto_decoder.train()
    elastic_tensor_decoder.train()

    N_EPOCHS = args.epochs
    now_step = 0

    min_vali_loss = float(checkpoint.get('loss', float('inf'))) if checkpoint else float('inf')
    start_epoch = int(checkpoint.get('epoch', -1)) + 1 if checkpoint else 0

    all_start_time = time.time()
    for epoch in range(start_epoch, N_EPOCHS):
        auto_encoder.train()
        auto_decoder.train()
        elastic_tensor_decoder.train()


        sampler.set_epoch(epoch)
        start_time = time.time()

        epoch_delta = 0
        epoch_loss = 0
        epoch_elastic_tensor_delta = 0
        epoch_elastic_tensor_loss = 0
        epoch_pro_delta = 0
        epoch_pro_loss = 0
        epoch_steps = 0


        for voxel_data, occup_data, property_data, elastic_tensor_data in dataloader:
            auto_encoder.train()
            auto_decoder.train()
            elastic_tensor_decoder.train()


            voxel_data = voxel_data.to(device)
            occup_data = occup_data.to(device)
            property_data = property_data.to(device)
            elastic_tensor_data = elastic_tensor_data.to(device)


            batch_size = voxel_data.shape[0]

            pts_sdf = occup_data

            coordinates, gt_occupancies = pts_sdf[..., 0:3], pts_sdf[..., -1]

            step_loss = 0
            step_elastic_tensor_loss = 0
            step_edr_loss = 0
            step_delta = 0
            step_l2_loss = 0
            step_tv_loss = 0
            step_sym_loss = 0
            step_pro_loss = 0

            optimizer.zero_grad()
            for _step in range(args.steps_per_batch):
                holoplane_latents = auto_encoder(voxel_data, property_data)

                # holoplane_latens 3, b, c, h, w
                pred_occup = auto_decoder(holoplane_latents, coordinates) # b, p, 1
                gt_occupancies = gt_occupancies.reshape((gt_occupancies.shape[0], gt_occupancies.shape[1], -1))
                delta = pred_occup - gt_occupancies
                rec_loss = (delta * delta).sum() / batch_size
                loss = rec_loss

                elastic_tensor_voxel_one_time = elastic_tensor_voxel.clone().repeat(batch_size, 1, 1)
                elastic_tensor_pred, pro_pred = elastic_tensor_decoder(holoplane_latents, elastic_tensor_voxel_one_time) # b, 64 * 64 * 64, 18, b, 3

                elastic_tensor_pred = elastic_tensor_pred.reshape((elastic_tensor_pred.shape[0], 64, 64, 64, 18))
                elastic_tensor_delta = elastic_tensor_pred - elastic_tensor_data
                elastic_tensor_loss = (elastic_tensor_delta * elastic_tensor_delta).sum() / batch_size / 18.0
                loss += elastic_tensor_loss
                step_elastic_tensor_loss += elastic_tensor_loss.item()

                pro_delta = pro_pred - property_data
                pro_loss = (pro_delta * pro_delta).sum() / batch_size / 3.0 * 20.0
                loss += pro_loss
                step_pro_loss += pro_loss.item()

                # Explicit density regulation
                if args.edr_val is not None and args.edr_val > 0:
                    smooth_loss = edr_loss(batch_size, holoplane_latents, auto_decoder, device, offset_distance=args.edr_val) * 5.0
                    loss += smooth_loss
                    step_edr_loss += smooth_loss.item()

                l2_loss = l2_reg(holoplane_latents) * 10.0
                tv_loss = tv_reg(holoplane_latents) * 50.0
                sym_loss = sym_loss_fun(holoplane_latents) * 5e-3

                step_l2_loss += l2_loss.item()
                step_tv_loss += tv_loss.item()
                step_sym_loss += sym_loss.item()

                loss += l2_loss
                loss += tv_loss
                loss += sym_loss

                step_loss += loss.item()
                step_delta += delta.abs().sum().item() / batch_size

                epoch_loss += loss.item()
                epoch_delta += delta.abs().sum().item()
                epoch_elastic_tensor_loss += elastic_tensor_loss.item()
                epoch_elastic_tensor_delta += elastic_tensor_delta.abs().sum().item() / (64 * 64 * 64 * 18)
                epoch_pro_loss += pro_loss.item()
                epoch_pro_delta += pro_delta.abs().sum().item() / 3
                epoch_steps += batch_size


                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite joint AE loss')
                (loss / args.steps_per_batch).backward()

            step_loss = step_loss / args.steps_per_batch
            step_delta = step_delta / args.steps_per_batch
            step_edr_loss = step_edr_loss / args.steps_per_batch
            step_l2_loss = step_l2_loss / args.steps_per_batch
            step_tv_loss = step_tv_loss / args.steps_per_batch
            step_sym_loss = step_sym_loss / args.steps_per_batch
            step_elastic_tensor_loss = step_elastic_tensor_loss / args.steps_per_batch
            step_pro_loss = step_pro_loss / args.steps_per_batch

            if local_rank == 0:
                writer.add_scalar('Loss', step_loss, now_step)
                writer.add_scalar('EDR Loss', step_edr_loss, now_step)
                writer.add_scalar('Delta', step_delta, now_step)
                writer.add_scalar('L2 Loss', step_l2_loss, now_step)
                writer.add_scalar('TV Loss', step_tv_loss, now_step)
                writer.add_scalar('Sym Loss', step_sym_loss, now_step)
                writer.add_scalar('Displacement Loss', step_elastic_tensor_loss, now_step)
                writer.add_scalar('Pro Loss', step_pro_loss, now_step)


            if args.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(auto_decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(auto_encoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(elastic_tensor_decoder.parameters(), args.grad_clip)

            optimizer.step()

            now_step += 1

        epoch_loss = epoch_loss / epoch_steps
        epoch_delta = epoch_delta / epoch_steps
        epoch_elastic_tensor_delta = epoch_elastic_tensor_delta / epoch_steps
        epoch_elastic_tensor_loss = epoch_elastic_tensor_loss / epoch_steps
        epoch_pro_delta = epoch_pro_delta / epoch_steps
        epoch_pro_loss = epoch_pro_loss / epoch_steps

        epoch_loss_per_point = epoch_loss / args.points_batch_size
        epoch_delta_per_point = epoch_delta / args.points_batch_size

        if local_rank == 0 and not epoch % args.log_every:
            print(f'Epoch {epoch}')
            print(f'del: {epoch_delta:.3f} loss: {epoch_loss:.3f}')
            print(f'elastic_tensor_delta: {epoch_elastic_tensor_delta:.3f} elastic_tensor_loss: {epoch_elastic_tensor_loss:.3f}')
            print(f'pro_delta: {epoch_pro_delta:.3f} pro_loss: {epoch_pro_loss:.3f}')
            print(f'losspt: {epoch_loss_per_point:.3f} delpt: {epoch_delta_per_point:.3f} time: {time.time() - start_time:.3f} time_per_step: {(time.time() - start_time) / epoch_steps:.3f}')

        if local_rank == 0 and args.vis_every > 0 and not epoch % args.vis_every:
            auto_encoder.eval()
            auto_decoder.eval()
            elastic_tensor_decoder.eval()

            with torch.no_grad():

                x = np.linspace(-1.0, 1.0, args.resolution)
                y = np.linspace(-1.0, 1.0, args.resolution)
                z = np.linspace(-1.0, 1.0, args.resolution)
                X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
                coords_mc = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1) # [resolution**3, 3]
                coords_mc = torch.Tensor(coords_mc).unsqueeze(0).to(device)

                random_id = random.randrange(len(dataset_vali))
                select_voxel, select_occup, select_label, select_elastic_tensor, select_name = dataset_vali[random_id]

                select_voxel = select_voxel.unsqueeze(0).to(device)
                select_occup = select_occup.unsqueeze(0).to(device)
                select_label = select_label.unsqueeze(0).to(device)

                holoplane_latents = auto_encoder.module(select_voxel, select_label)
                occup_pred = auto_decoder.module(holoplane_latents, select_occup[..., 0:3])

                coords_mc_one_time = coords_mc.clone()
                mc_occup = torch.cat([auto_decoder.module(holoplane_latents, chunk) for chunk in coords_mc_one_time.split(args.vis_chunk_size, dim=1)], dim=1)
                # gradient_magnitudes[gradient_magnitudes > 5] = 5
                mc_vis = mc_occup.detach().cpu()
                mc_occup = mc_occup.squeeze(-1).detach().cpu().reshape(X.shape).numpy()

                visualize_holoplane(epoch, 0, holoplane_latents, writer)
                visualize_occupancy(epoch, 0, select_occup[..., 0:3], occup_pred, select_occup[..., -1], writer)
                visualize_occupancy_error(epoch, 0, select_occup[..., 0:3], occup_pred, select_occup[..., -1], writer)
                visualize_sdf_clip(epoch, 0, mc_vis, writer, select_name + "_pr")
                select_voxel = select_voxel.cpu()
                # visualize_sdf_clip(epoch, 0, select_voxel, writer, select_name + "_gt")
            torch.cuda.empty_cache()

        if not epoch % args.val_every:
            auto_encoder.eval()
            auto_decoder.eval()
            elastic_tensor_decoder.eval()

            with torch.no_grad():
                rec_delta_all = 0
                rec_loss_all = 0
                conms_delta_all = 0
                conms_loss_all = 0
                pro_delta_all = 0
                pro_loss_all = 0

                total_cnt = 0
                for voxel_data, occup_data, property_data, elastic_tensor_data, name_data in dataloader_vali:

                    voxel_data = voxel_data.to(device)
                    occup_data = occup_data.to(device)
                    property_data = property_data.to(device)
                    elastic_tensor_data = elastic_tensor_data.to(device)

                    batch_size = voxel_data.shape[0]

                    pts_sdf = occup_data
                    coordinates, gt_occupancies = pts_sdf[..., 0:3], pts_sdf[..., -1]

                    holoplane_latents = auto_encoder.module(voxel_data, property_data)
                    pred_occup = auto_decoder.module(holoplane_latents, coordinates)
                    gt_occupancies = gt_occupancies.reshape((gt_occupancies.shape[0], gt_occupancies.shape[1], -1))
                    rec_delta = pred_occup - gt_occupancies
                    rec_delta_p = rec_delta.abs().sum() / args.points_batch_size
                    rec_loss = (rec_delta * rec_delta).sum() / args.points_batch_size

                    elastic_tensor_voxel_one_time = elastic_tensor_voxel.clone().repeat(batch_size, 1, 1)
                    elastic_tensor_pred, pro_pred = elastic_tensor_decoder.module(holoplane_latents, elastic_tensor_voxel_one_time) # b, 64 * 64 * 64, 18, b, 3

                    elastic_tensor_pred = elastic_tensor_pred.reshape((elastic_tensor_pred.shape[0], 64, 64, 64, 18))
                    elastic_tensor_delta = elastic_tensor_pred - elastic_tensor_data
                    elastic_tensor_delta_p = elastic_tensor_delta.abs().sum() / (64 * 64 * 64 * 18)
                    elastic_tensor_loss = (elastic_tensor_delta * elastic_tensor_delta).sum() / (64 * 64 * 64 * 18)

                    pro_delta = pro_pred - property_data
                    pro_delta_p = pro_delta.abs().sum() / 3
                    pro_loss = (pro_delta * pro_delta).sum() / 3.0

                    rec_delta_all += rec_delta_p.item()
                    rec_loss_all += rec_loss.item()
                    conms_delta_all += elastic_tensor_delta_p.item()
                    conms_loss_all += elastic_tensor_loss.item()
                    pro_delta_all += pro_delta_p.item()
                    pro_loss_all += pro_loss.item()
                    total_cnt += batch_size

                to_reduce = torch.Tensor([rec_delta_all, rec_loss_all, conms_delta_all, conms_loss_all, pro_delta_all, pro_loss_all, total_cnt]).to(device)
                torch.distributed.all_reduce(to_reduce)

                if local_rank == 0:
                    rec_delta_all, rec_loss_all, conms_delta_all, conms_loss_all, pro_delta_all, pro_loss_all, total_cnt = to_reduce.tolist()
                    rec_delta_all /= total_cnt
                    rec_loss_all /= total_cnt
                    conms_delta_all /= total_cnt
                    conms_loss_all /= total_cnt
                    pro_delta_all /= total_cnt
                    pro_loss_all /= total_cnt
                    now_loss = rec_loss_all + conms_loss_all + pro_loss_all

                    print("---------------------")
                    print(f'Validation epoch: {epoch}')
                    print(f'Validation cnt: {total_cnt}')
                    print(f'Validation rec delta: {rec_delta_all}')
                    print(f'Validation rec loss: {rec_loss_all}')
                    print(f'Validation con delta: {conms_delta_all}')
                    print(f'Validation con loss: {conms_loss_all}')
                    print(f'Validation pro delta: {pro_delta_all}')
                    print(f'Validation pro loss: {pro_loss_all}')
                    print(f'Validation total loss: {now_loss}')

                    print("---------------------")

                    writer.add_scalar('Validation Delta', rec_delta_all, epoch)
                    writer.add_scalar('Validation Loss', rec_loss_all, epoch)
                    writer.add_scalar('Validation U Delta', conms_delta_all, epoch)
                    writer.add_scalar('Validation U Loss', conms_loss_all, epoch)
                    writer.add_scalar('Validation Pro Delta', pro_delta_all, epoch)
                    writer.add_scalar('Validation Pro Loss', pro_loss_all, epoch)


                    if now_loss < min_vali_loss and not epoch % args.save_every and ckpt_path:
                        min_vali_loss = now_loss
                        print(f'Saving checkpoint at epoch {epoch}')
                        torch.save({
                            'epoch': epoch,
                            'decoder_state_dict': auto_decoder.module.state_dict(),
                            'encoder_state_dict': auto_encoder.module.state_dict(),
                            'elastic_tensor_decoder_state_dict': elastic_tensor_decoder.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'decoder_kwargs': {'use_tanh': args.use_tanh},
                            'loss': now_loss,
                        }, f'{ckpt_path}/model_epoch_{epoch}_loss_{now_loss}.pt')
        torch.cuda.empty_cache()

    if local_rank == 0:
        writer.close()
    torch.distributed.destroy_process_group()

def setRandomSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
