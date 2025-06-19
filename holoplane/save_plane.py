

'''
Train auto-decoder.

'''


from concurrent.futures import ThreadPoolExecutor
import os
import argparse
import random
import numpy as np
import torch
from dataset import VoxelOccupancyDataset
from triplane_ae import TriplaneDecoder, TriplaneEncoder
from torch.utils.data import DataLoader
# import wandb
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter

def existsOrMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def main():
    parser = argparse.ArgumentParser(description='Train auto-decoder.')
    parser.add_argument('--dataset_voxel', type=str,
                    help='directory to voxels', required=True)
    parser.add_argument('--dataset_occup', type=str,
                    help='directory to occupancies', required=True)
    parser.add_argument('--dataset_plane', type=str,
                    help='directory to holoplane', required=True)
    parser.add_argument('--dataset_list', type=str,
                    help='train filelist', required=True)
    parser.add_argument('--repeat', type=int, default=4, required=False)

    parser.add_argument('--log_dir', type=str,
                    help='directory to log', required=True)
    parser.add_argument('--batch_size', type=int, default=1, required=False,
                    help='number of scenes per batch')
    parser.add_argument('--points_batch_size', type=int, default=100000, required=False,
                    help='number of points per scene, nss and uniform combined')
    
    parser.add_argument('--log_every', type=int, default=1, required=False)
    parser.add_argument('--val_every', type=int, default=32, required=False)
    parser.add_argument('--vis_every', type=int, default=64, required=False)
    parser.add_argument('--save_every', type=int, default=256, required=False)
    
    parser.add_argument('--load_ckpt_path', type=str, default=None, required=False,
                    help='checkpoint to continue training from')
    parser.add_argument('--checkpoint_path', type=str, default='./ckpts', required=False,
                    help='where to save model checkpoints')
    
    parser.add_argument('--resolution', type=int, default=128, required=False,
                    help='triplane resolution')
    parser.add_argument('--channels', type=int, default=32, required=False,
                    help='triplane depth')
    parser.add_argument('--aggregate_fn', type=str, default='prod', required=False,
                    help='function for aggregating triplane features')
    parser.add_argument('--seed', type=int, default=3407)
    
    args = parser.parse_args()
    
    assert args.load_ckpt_path is not None, 'Please provide a checkpoint to load from.'
    
    setRandomSeed(args.seed)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    # wandb.init(project="triplane-autodecoder")
    
    main_logs = "../logs"
    existsOrMkdir(main_logs)
    tb_path = f'{main_logs}/{args.log_dir}'
    existsOrMkdir(tb_path)
    writer = SummaryWriter(tb_path, "base")
    
    triplane_file_folder = args.dataset_plane


    dataset_vali = VoxelOccupancyDataset(resolution=args.resolution, file_list_name=args.dataset_list, voxel_dir=args.dataset_voxel, occup_dir=args.dataset_occup, points_batch_size=args.points_batch_size, return_full=True, device=device)
    dataloader = DataLoader(dataset_vali, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    auto_encoder = TriplaneEncoder(resolution=args.resolution, in_channels=1, out_channels=args.channels, device=device).to(device)
    # auto_decoder = TriplaneDecoder(resolution=args.resolution, channels=args.channels, aggregate_fn=args.aggregate_fn, use_tanh=args.use_tanh).to(device)
    
    checkpoint = torch.load(args.load_ckpt_path)
    auto_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    # auto_decoder.load_state_dict(checkpoint['decoder_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Loaded checkpoint from {args.load_ckpt_path}. Resuming training from epoch {epoch} with loss {loss}...')

    auto_encoder.eval()
    
    def make_symmetric(now_tri):
        # now_tri_shape 3, 32, 64, 64
        plane_flipped_x  = np.flip(now_tri, axis=2)
        plane_flipped_y  = np.flip(now_tri, axis=3)
        plane_flipped_xy = np.flip(now_tri, axis=(2, 3))
        
        symmetrized = (now_tri + plane_flipped_x + plane_flipped_y + plane_flipped_xy) / 4
        return symmetrized
    
    def save_triplane_latents(i, triplane_latents_np, name_data, triplane_file_folder):
        save_file_name = f"{triplane_file_folder}/{name_data[i]}.npy"
        now_tri = triplane_latents_np[i]
        symmetrized = make_symmetric(now_tri)
        if not os.path.exists(save_file_name):
            np.save(save_file_name, symmetrized)

    all_start_time = time.time()
    
    repeat_times = args.repeat
    
    for voxel_data, occup_data, name_data in tqdm(dataloader):
        all_existed = True
        for name in name_data:
            if not os.path.exists(f"{triplane_file_folder}/{name}.npy"):
                all_existed = False
                break
        if all_existed:
            continue
        start_time = time.time()

        voxel_data = voxel_data.to(device)
        occup_data = occup_data.to(device)
        
        with torch.no_grad():
            batch_size = voxel_data.shape[0]
            triplane_latents_repeated = np.zeros((repeat_times, batch_size, 3, 32, 64, 64))
            for i in range(repeat_times):
                triplane_latents = auto_encoder(voxel_data)
                triplane_latents_np = np.stack([
                    triplane_latent.detach().cpu().numpy() for triplane_latent in triplane_latents
                ])  # (3, batch_size, 32, 64, 64)
                triplane_latents_np = np.swapaxes(triplane_latents_np, 0, 1)  # (batch_size, 3, 32, 64, 64)
                triplane_latents_repeated[i] = triplane_latents_np
        
        triplane_latents_avg = np.mean(triplane_latents_repeated, axis=0)

        with ThreadPoolExecutor(max_workers=8) as executor: 
            for i in range(triplane_latents_avg.shape[0]):
                executor.submit(save_triplane_latents, i, triplane_latents_avg, name_data, triplane_file_folder)
        '''
        triplane_latents = np.stack(triplane_latents)
        # print(triplane_latents.shape)

        save_file_name = f"{triplane_file_folder}/{name_data}.npy"
        np.save(save_file_name, triplane_latents)
        '''
        torch.cuda.empty_cache()
        
    writer.close()

def setRandomSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
if __name__ == "__main__":
    main()

