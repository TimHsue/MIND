

'''
Train auto-decoder.

'''

import os
import argparse
import random
import numpy as np
import torch
from dataset import VoxelOccupancyDataset
from holoplane_ae import TriplaneDecoder, TriplaneEncoder, ConstitutiveMatrixDecoder
from torch.utils.data import DataLoader
# import wandb
import time
from tensorboardX import SummaryWriter
from regularization import edr_loss, l2_reg, tv_reg, sym_loss_fun
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vis_tools import visualize_mesh, visualize_triplane, visualize_occupancy, visualize_occupancy_error, visualize_voxel, visualize_sdf_clip, visualize_sdf_norm_clip


def existsOrMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def load_state_dict(model, state_dict):
    for key, value in state_dict.items():
        if 'module' in key:
            key = key.replace('module.', '')
        try:
            model.state_dict()[key].copy_(value)
        except Exception as e:
            print(f'Load dict error: {e}')
            print(f'Key: {key}')
            print('-----------------')
            continue



def main():
    parser = argparse.ArgumentParser(description='Train auto-decoder.')
    parser.add_argument('--dataset_voxel', type=str,
                    help='directory to voxels', required=True)
    parser.add_argument('--dataset_occup', type=str,
                    help='directory to occupancies', required=True)
    parser.add_argument('--dataset_consm', type=str,
                    help='directory to constitutive matrix', required=True)
    parser.add_argument('--dataset_list', type=str,
                    help='train filelist', required=True)
    parser.add_argument('--dataset_list_vali', type=str,
                    help='vali filelist', required=True)
    parser.add_argument('--dataset_pro', type=str,
                    help='train filelist', required=True)
    parser.add_argument('--dataset_pro_vali', type=str,
                    help='vali filelist', required=True)
    parser.add_argument('--log_dir', type=str,
                    help='directory to log', required=True)
        
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='number of scenes per batch')
    parser.add_argument('--points_batch_size', type=int, default=100000, required=False,
                    help='number of points per scene, nss and uniform combined')
    parser.add_argument('--log_every', type=int, default=1, required=False)
    parser.add_argument('--val_every', type=int, default=1, required=False)
    parser.add_argument('--vis_every', type=int, default=1, required=False)
    parser.add_argument('--save_every', type=int, default=4, required=False)
    
    parser.add_argument('--load_ckpt_path', type=str, default=None, required=False,
                    help='checkpoint to continue training from')
    parser.add_argument('--checkpoint_path', type=str, default='./ckpts', required=False,
                    help='where to save model checkpoints')
    parser.add_argument('--only_decoder', type=bool, default=False, required=False,
                    help='train only decoder')
    
    parser.add_argument('--resolution', type=int, default=128, required=False,
                    help='holoplane resolution')
    parser.add_argument('--channels', type=int, default=32, required=False,
                    help='holoplane depth')
    parser.add_argument('--aggregate_fn', type=str, default='prod', required=False,
                    help='function for aggregating holoplane features')
    
    parser.add_argument('--subset_size', type=int, default=None, required=False,
                    help='size of the dataset subset if we\'re training on a subset')
    parser.add_argument('--steps_per_batch', type=int, default=1, required=False,
                    help='If specified, how many GD steps to run on a batch before moving on. To address I/O stuff.')
    parser.add_argument('--edr_val', type=float, default=None, required=False,
                    help='If specified, use explicit density regularization with the specified offset distance value.')
    parser.add_argument('--triplet_margin', type=float, default=None, required=False,
                    help='If specified, use explicit density regularization with the specified offset distance value.')
    parser.add_argument('--use_tanh', default=False, required=False, action='store_true',
                    help='Whether to use tanh to clamp holoplanes to [-1, 1].')
    
    parser.add_argument('--e_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--p_lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--grad_clip', type=float, default=1.)
    
    args = parser.parse_args()
    
    # use torch.distributed.launch to set the local rank automatically
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    # initialize the process group
    print("preparing process group...")
    torch.distributed.init_process_group("nccl", rank=local_rank, world_size=world_size)
    
    setRandomSeed(args.seed + local_rank)
    # create the model and move it to the local GPU
    device = torch.device("cuda", local_rank)
    print("device locked ", local_rank)
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    torch.set_num_threads(16)
    
    # device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # wandb.init(project="holoplane-autodecoder")
    ckpt_path = None
    if local_rank == 0:
        main_logs = "../logs"
        existsOrMkdir(main_logs)
        tb_path = f'{main_logs}/{args.log_dir}'
        existsOrMkdir(tb_path)
        writer = SummaryWriter(tb_path, "base")
        if args.checkpoint_path:
            ckpt_path = f'{main_logs}/{args.log_dir}/{args.checkpoint_path}' 
            os.makedirs(ckpt_path, exist_ok=True)

    checkpoint = None
    if args.load_ckpt_path:
        checkpoint = torch.load(args.load_ckpt_path, map_location="cpu")
        print(f'Loaded checkpoint from {args.load_ckpt_path}. Resuming training from epoch {checkpoint["epoch"]} with loss {checkpoint["loss"]}...')
        
    dataset = VoxelOccupancyDataset(
        resolution=args.resolution, 
        file_list_name=args.dataset_list, 
        voxel_dir=args.dataset_voxel, 
        occup_dir=args.dataset_occup, 
        consm_dir=args.dataset_consm,
        dataset_pro=args.dataset_pro,
        points_batch_size=args.points_batch_size, 
        return_name=False,  
        return_property=True,
        return_consm=True,
        device=device)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, sampler=sampler)
    
    dataset_vali = VoxelOccupancyDataset(
        resolution=args.resolution, 
        file_list_name=args.dataset_list_vali, 
        voxel_dir=args.dataset_voxel, 
        occup_dir=args.dataset_occup, 
        consm_dir=args.dataset_consm,
        dataset_pro=args.dataset_pro_vali,
        points_batch_size=args.points_batch_size, 
        return_name=True, 
        return_consm=True,
        return_property=True,
        device=device)
    sampler_vali = torch.utils.data.distributed.DistributedSampler(dataset_vali)
    dataloader_vali = DataLoader(dataset_vali, batch_size=args.batch_size, shuffle=False, num_workers=8, sampler=sampler_vali)
    
    auto_encoder = TriplaneEncoder(
        resolution=args.resolution, 
        in_channels=1, 
        out_channels=args.channels, 
        label_dim=3,  
        label_scale=[10, 10, 10],
        device=device).to(device)
    
    auto_decoder = TriplaneDecoder(
        channels=args.channels, 
        aggregate_fn=args.aggregate_fn, 
        use_coord=False,
        use_tanh=args.use_tanh).to(device)

    consm_decoder = ConstitutiveMatrixDecoder(
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
        if 'consm_decoder_state_dict' in checkpoint:
            load_state_dict(consm_decoder, checkpoint['consm_decoder_state_dict'])
    

    auto_encoder = torch.nn.parallel.DistributedDataParallel(auto_encoder, device_ids=[local_rank], output_device=local_rank)
    auto_decoder = torch.nn.parallel.DistributedDataParallel(auto_decoder, device_ids=[local_rank], output_device=local_rank)
    consm_decoder = torch.nn.parallel.DistributedDataParallel(consm_decoder, device_ids=[local_rank], output_device=local_rank)

    if args.only_decoder:
        optimizer = torch.optim.Adam(auto_decoder.parameters(), lr=args.d_lr)
    else:
        optimizer = torch.optim.Adam([
                        {'params': auto_encoder.parameters(), 'lr': args.e_lr},
                        {'params': auto_decoder.parameters(), 'lr': args.d_lr},
                        {'params': consm_decoder.parameters(), 'lr': args.p_lr}
                    ], betas=(0.9, 0.999))

    # update lr
    if args.only_decoder:
        optimizer.param_groups[0]['lr'] = args.d_lr
    else:
        optimizer.param_groups[0]['lr'] = args.e_lr
        optimizer.param_groups[1]['lr'] = args.d_lr
        optimizer.param_groups[2]['lr'] = args.p_lr
        
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    consm_x = np.linspace(-0.8, 0.8, 64)
    consm_y = np.linspace(-0.8, 0.8, 64)
    consm_z = np.linspace(-0.8, 0.8, 64)
    consm_X, consm_Y, consm_Z = np.meshgrid(consm_x, consm_y, consm_z)
    consm_voxel = np.stack([consm_X.ravel(), consm_Y.ravel(), consm_Z.ravel()], axis=-1)
    consm_voxel = torch.Tensor(consm_voxel).unsqueeze(0).to(device) # 1, 64 * 64 * 64, 3

    auto_encoder.train()
    auto_decoder.train()
    consm_decoder.train()
    
    N_EPOCHS = 3000000
    now_step = 0
    
    min_vali_loss = 100000
    
    all_start_time = time.time()
    for epoch in range(N_EPOCHS):


        sampler.set_epoch(epoch)      
        sampler_vali.set_epoch(epoch)      
        start_time = time.time()
        
        epoch_delta = 0
        epoch_loss = 0 
        epoch_consm_delta = 0 
        epoch_consm_loss = 0
        epoch_pro_delta = 0
        epoch_pro_loss = 0
        epoch_steps = 0
        
        
        for voxel_data, occup_data, property_data, consm_data in dataloader:
            auto_encoder.train()
            auto_decoder.train()
            consm_decoder.train()


            voxel_data = voxel_data.to(device)
            occup_data = occup_data.to(device)
            property_data = property_data.to(device)
            consm_data = consm_data.to(device)


            batch_size = voxel_data.shape[0]
            
            pts_sdf = occup_data

            coordinates, gt_occupancies = pts_sdf[..., 0:3], pts_sdf[..., -1]

            step_loss = 0
            step_consm_loss = 0
            step_edr_loss = 0
            step_delta = 0
            step_l2_loss = 0
            step_tv_loss = 0
            step_sym_loss = 0
            step_pro_loss = 0

            for _step in range(args.steps_per_batch):     
                if args.only_decoder:
                    auto_encoder.eval()
                    with torch.no_grad():
                        triplane_latents = auto_encoder(voxel_data, property_data)
                        # triplane_latents = [triplane_latent + torch.randn_like(triplane_latent) * 0.008 for triplane_latent in triplane_latents]
                else:
                    triplane_latents = auto_encoder(voxel_data, property_data)

                # triplane_latens 3, b, c, h, w
                triplane_latents = auto_encoder(voxel_data, property_data) # b, 3, 32, 64, 64
                pred_occup = auto_decoder(triplane_latents, coordinates) # b, p, 1
                gt_occupancies = gt_occupancies.reshape((gt_occupancies.shape[0], gt_occupancies.shape[1], -1))
                delta = pred_occup - gt_occupancies
                rec_loss = (delta * delta).sum() / batch_size
                loss = rec_loss

                consm_voxel_one_time = consm_voxel.clone().repeat(batch_size, 1, 1)
                consm_pred, pro_pred = consm_decoder(triplane_latents, consm_voxel_one_time) # b, 64 * 64 * 64, 18, b, 3

                consm_pred = consm_pred.reshape((consm_pred.shape[0], 64, 64, 64, 18))
                consm_delta = consm_pred - consm_data
                consm_loss = (consm_delta * consm_delta).sum() / batch_size / 18.0
                loss += consm_loss
                step_consm_loss += consm_loss.item()

                pro_delta = pro_pred - property_data
                pro_loss = (pro_delta * pro_delta).sum() / batch_size / 3.0 * 20.0
                loss += pro_loss
                step_pro_loss += pro_loss.item()

                # Explicit density regulation
                if args.edr_val is not None and args.edr_val > 0:
                    smooth_loss = edr_loss(batch_size, triplane_latents, auto_decoder, device, offset_distance=args.edr_val) * 5.0
                    loss += smooth_loss
                    step_edr_loss += smooth_loss.item()
                
                l2_loss = l2_reg(triplane_latents) * 10.0
                tv_loss = tv_reg(triplane_latents) * 50.0
                sym_loss = sym_loss_fun(triplane_latents) * 5e-3
                
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
                epoch_consm_loss += consm_loss.item()
                epoch_consm_delta += consm_delta.abs().sum().item() / (64 * 64 * 64 * 18)
                epoch_pro_loss += pro_loss.item()
                epoch_pro_delta += pro_delta.abs().sum().item() / 3
                epoch_steps += batch_size


                optimizer.zero_grad()
                loss.backward()

            step_loss = step_loss / args.steps_per_batch
            step_delta = step_delta / args.steps_per_batch
            step_edr_loss = step_edr_loss / args.steps_per_batch
            step_l2_loss = step_l2_loss / args.steps_per_batch
            step_tv_loss = step_tv_loss / args.steps_per_batch
            step_sym_loss = step_sym_loss / args.steps_per_batch
            step_consm_loss = step_consm_loss / args.steps_per_batch
            step_pro_loss = step_pro_loss / args.steps_per_batch
            
            if local_rank == 0:
                writer.add_scalar('Loss', step_loss, now_step)
                writer.add_scalar('EDR Loss', step_edr_loss, now_step)
                writer.add_scalar('Delta', step_delta, now_step)
                writer.add_scalar('L2 Loss', step_l2_loss, now_step)
                writer.add_scalar('TV Loss', step_tv_loss, now_step)
                writer.add_scalar('Sym Loss', step_sym_loss, now_step)
                writer.add_scalar('Consm Loss', step_consm_loss, now_step)
                writer.add_scalar('Pro Loss', step_pro_loss, now_step)
                
            
            if args.grad_clip >= 0:
                torch.nn.utils.clip_grad_norm_(auto_decoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(auto_encoder.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(consm_decoder.parameters(), args.grad_clip)
                
            optimizer.step()

            now_step += 1
        
        epoch_loss = epoch_loss / epoch_steps
        epoch_delta = epoch_delta / epoch_steps
        epoch_consm_delta = epoch_consm_delta / epoch_steps
        epoch_consm_loss = epoch_consm_loss / epoch_steps
        epoch_pro_delta = epoch_pro_delta / epoch_steps
        epoch_pro_loss = epoch_pro_loss / epoch_steps
        
        epoch_loss_per_point = epoch_loss / args.points_batch_size
        epoch_delta_per_point = epoch_delta / args.points_batch_size
        
        if local_rank == 0 and not epoch % args.log_every: 
            print(f'Epoch {epoch}')
            print(f'del: {epoch_delta:.3f} loss: {epoch_loss:.3f}')
            print(f'consm_delta: {epoch_consm_delta:.3f} consm_loss: {epoch_consm_loss:.3f}')
            print(f'pro_delta: {epoch_pro_delta:.3f} pro_loss: {epoch_pro_loss:.3f}')
            print(f'losspt: {epoch_loss_per_point:.3f} delpt: {epoch_delta_per_point:.3f} time: {time.time() - start_time:.3f} time_per_step: {(time.time() - start_time) / epoch_steps:.3f}') 
        
        if local_rank == 0 and not epoch % args.vis_every:
            auto_encoder.eval()
            auto_decoder.eval()
            consm_decoder.eval()
        
            with torch.no_grad():

                x = np.linspace(-1.0, 1.0, args.resolution)
                y = np.linspace(-1.0, 1.0, args.resolution)
                z = np.linspace(-1.0, 1.0, args.resolution)
                X, Y, Z = np.meshgrid(x, y, z)
                coords_mc = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1) # 64 * 64 * 64, 3
                coords_mc = torch.Tensor(coords_mc).unsqueeze(0).to(device)
        
                random_id = random.randint(0, len(dataset_vali))
                select_voxel, select_occup, select_label, select_consm, select_name = dataset_vali[random_id]
                
                select_voxel = select_voxel.unsqueeze(0).to(device)
                select_occup = select_occup.unsqueeze(0).to(device)
                select_label = select_label.unsqueeze(0).to(device)

                triplane_latents = auto_encoder(select_voxel, select_label)
                occup_pred = auto_decoder(triplane_latents, select_occup[..., 0:3])
                
                coords_mc_one_time = coords_mc.clone()
                mc_occup = auto_decoder(triplane_latents, coords_mc_one_time)
                # mc_occup_grad = torch.autograd.grad(mc_occup, coords_mc_one_time, torch.ones_like(mc_occup), create_graph=True)[0].detach()
                # gradient_magnitudes = mc_occup_grad.norm(dim=-1)
                # gradient_magnitudes[gradient_magnitudes > 5] = 5
                mc_vis = mc_occup.detach().cpu()
                mc_occup = mc_occup.squeeze(-1).detach().cpu().reshape(X.shape).numpy()
            
                visualize_triplane(epoch, 0, triplane_latents, writer)
                visualize_occupancy(epoch, 0, select_occup[..., 0:3], occup_pred, select_occup[..., -1], writer)
                visualize_occupancy_error(epoch, 0, select_occup[..., 0:3], occup_pred, select_occup[..., -1], writer)
                visualize_sdf_clip(epoch, 0, mc_vis, writer, select_name + "_pr")
                select_voxel = select_voxel.cpu()
                # visualize_sdf_clip(epoch, 0, select_voxel, writer, select_name + "_gt")
                # visualize_sdf_norm_clip(epoch, 0, gradient_magnitudes, writer)
            torch.cuda.empty_cache()
        
        if not epoch % args.val_every:
            auto_encoder.eval()
            auto_decoder.eval()
            consm_decoder.eval()

            with torch.no_grad():
                rec_delta_all = 0
                rec_loss_all = 0
                conms_delta_all = 0
                conms_loss_all = 0
                pro_delta_all = 0
                pro_loss_all = 0

                total_cnt = 0  
                for voxel_data, occup_data, property_data, consm_data, name_data in dataloader_vali:

                    voxel_data = voxel_data.to(device)
                    occup_data = occup_data.to(device)
                    property_data = property_data.to(device)
                    consm_data = consm_data.to(device)
                    
                    batch_size = voxel_data.shape[0]
                    
                    pts_sdf = occup_data
                    coordinates, gt_occupancies = pts_sdf[..., 0:3], pts_sdf[..., -1]
                    
                    triplane_latents = auto_encoder(voxel_data, property_data)
                    pred_occup = auto_decoder(triplane_latents, coordinates)
                    gt_occupancies = gt_occupancies.reshape((gt_occupancies.shape[0], gt_occupancies.shape[1], -1))
                    rec_delta = pred_occup - gt_occupancies
                    rec_delta_p = rec_delta.abs().sum() / args.points_batch_size
                    rec_loss = (rec_delta * rec_delta).sum() / args.points_batch_size

                    consm_voxel_one_time = consm_voxel.clone().repeat(batch_size, 1, 1)
                    consm_pred, pro_pred = consm_decoder(triplane_latents, consm_voxel_one_time) # b, 64 * 64 * 64, 18, b, 3

                    consm_pred = consm_pred.reshape((consm_pred.shape[0], 64, 64, 64, 18))
                    consm_delta = consm_pred - consm_data
                    consm_delta_p = consm_delta.abs().sum() / (64 * 64 * 64 * 18)
                    consm_loss = (consm_delta * consm_delta).sum() / (64 * 64 * 64 * 18)

                    pro_delta = pro_pred - property_data
                    pro_delta_p = pro_delta.abs().sum() / 3
                    pro_loss = (pro_delta * pro_delta).sum() / 3.0
                
                    rec_delta_all += rec_delta_p.item()
                    rec_loss_all += rec_loss.item()
                    conms_delta_all += consm_delta_p.item()
                    conms_loss_all += consm_loss.item()
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
                            'consm_decoder_state_dict': consm_decoder.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
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

