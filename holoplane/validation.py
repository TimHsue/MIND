

'''
Train auto-decoder.

'''


import json
import os
import argparse
import pickle
import random
import numpy as np
import torch
import tqdm
import trimesh
from triplane_ae import TriplaneDecoder
# import wandb
import time
from tensorboardX import SummaryWriter
from skimage import measure
import click
import dnnlib
from torch_utils import distributed as dist

import numpy as np 
from homogenization import do_voxel_homogenization,do_voxel_homogenization_full
import time
import torch
from tqdm import trange
import argparse
import os
import skimage.measure as sm
from scipy.ndimage import label


def find_largest_component(voxel):
    labels, num = sm.label(voxel, return_num=True, connectivity=1)
    # print("--------------------")
    # print(num)
    n = voxel.size
    label_counts = np.bincount(labels.flat)[1:]
    qualified = False
    if label_counts.size > 0:  
        qualified = label_counts.max() / label_counts.sum() > 0.9
        # print(label_counts.max(), label_counts.sum())
        # print("####################")
        if qualified:
            largest_label = np.argmax(label_counts) + 1
            largest_component = (labels == largest_label)
            return largest_component * voxel, qualified
        else:
            return voxel, qualified
    return voxel, qualified

def analysis_macro_properties_1(CH):
    SH = np.linalg.inv(CH)
    E=(1./SH[0,0]+1./SH[1,1]+1./SH[2,2])/3
    # E = 1./SH[0,0]
    v=-(SH[1,0]/SH[0,0]+SH[2,0]/SH[0,0]+SH[0,1]/SH[1,1]+SH[2,1]/SH[1,1]+SH[0,2]/SH[2,2]+SH[1,2]/SH[2,2])/6
    # nu = -SH[1,0]/SH[0,0] = -SH[1,0] * E
    # SH[1, 0] = - nu / E
    G=1/3*(1/SH[3,3]+1/SH[4,4]+1/SH[5,5])
    # G = 1 / SH[3, 3]
    
    return E,v,G

def check_nu35_single(voxel, names):
    # print(voxel.shape)
    voxel, q = find_largest_component(voxel)

    if not q:
        raise ValueError('Not qualified')

    vf = voxel.sum() / voxel.size
    C, error = do_voxel_homogenization(voxel, nu=0.35)
    CH = C.detach().cpu().numpy()
    torch.cuda.empty_cache()
    E, nu, G = analysis_macro_properties_1(CH)
    
    if error > 0.01:
        q = False
        print(f'error: {error}, res: {voxel.shape}, {names if names is not None else ""}')
        # C, error = do_voxel_homogenization_full(voxel, nu=0.35)
        # torch.cuda.empty_cache()
        # CH = C.detach().cpu().numpy()
        # E, nu, G = analysis_macro_properties_1(CH)
    return E, nu, G, vf, CH, error

def check_nu35(occups, names=None):
    size = len(occups)
    E=np.zeros((size,))
    nu=np.zeros_like(E)
    G=np.zeros_like(E)
    vf=np.zeros_like(E)
    err=np.zeros_like(E)
    CH=np.zeros((size,6,6))
    qualified=np.ones((size,), dtype=np.bool_)
    
    for i in tqdm.tqdm(range(size), total=size, desc='Checking Stiffness', disable=(dist.get_rank() != 0), dynamic_ncols=True, leave=False):
        # print(lists[i])
        try:
            E[i],nu[i],G[i],vf[i],CH[i],err[i]=check_nu35_single(occups[i], names[i] if names is not None else None)
        except Exception as e:
            # print(e)
            qualified[i]=False
    return E, nu, G, vf, CH, qualified, err


def existsOrMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--target_folder' ,          help='Target triplane folder', metavar='DIR',                            type=str, required=True)
@click.option('--log_dir',                 help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--batch_size',              help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--aggregate_fn',            help='function for aggregating triplane features', metavar='STR',        type=str, default='cat', show_default=True)
@click.option('--channels',                help='triplane depth', metavar='INT',                                    type=click.IntRange(min=1), default=32)
@click.option('--resolution',              help='triplane res', metavar='INT',                                      type=click.IntRange(min=1), default=128)
@click.option('--sym',                     help='triplane sym', metavar='INT',                                      type=click.IntRange(min=1), default=48)

@click.option('--seed',                    help='seed', metavar='INT',                                              type=click.IntRange(min=1), default=3407)
 
def main(network_pkl, target_folder, log_dir, batch_size, aggregate_fn, channels, resolution, sym, seed, device=torch.device('cuda'),  **sampler_kwargs):
    dist.init()
    max_batch_size = batch_size

    all_files = os.listdir(target_folder)
    all_files.sort()
    
    num_batches = ((len(all_files) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(all_files, num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    setRandomSeed(seed)
    
    auto_decoder = pickle.load(open(network_pkl, "rb"))['dec']
    auto_decoder.eval().requires_grad_(False).to(device)
    
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)

    
    x = np.linspace(-1.0, 1.0, resolution)
    y = np.linspace(-1.0, 1.0, resolution)
    z = np.linspace(-1.0, 1.0, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    coords_mc = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1) # shape (resolution^3, 3)
    coords_mc = torch.Tensor(coords_mc).unsqueeze(0).to(device)
    
    dist.print0(f'Generating {len(all_files)} results to "{log_dir}"...')
    for batch_files in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0), desc='Batching', dynamic_ncols=True, leave=True):
        # print(f'batch_files: {batch_files}')
        # torch.distributed.barrier()
        
        triplanes = []
        for name in batch_files:
            with open(f'{target_folder}/{name}', 'rb') as f:
                
                triplane_latent = np.load(f)
                # print(file, 'loaded')
                triplane_latent = torch.Tensor(triplane_latent).to(device)
                triplane_latent.reshape(96, triplane_latent.shape[-1], triplane_latent.shape[-1])
                if triplane_latent.shape[-1] == 32:
                    y_flip = torch.flip(triplane_latent, dims=[-1]) # shape (96, 32, 32)
                    cat_latent = torch.cat([triplane_latent, y_flip], dim=-1) # shape (96, 32, 64)
                    x_flip = torch.flip(cat_latent, dims=[-2]) # shape (96, 32, 64)
                    cat_latent = torch.cat([cat_latent, x_flip], dim=-2) # shape (96, 64, 64)
                    triplanes.append(cat_latent)

        # print(name_mask)           
        triplanes = torch.stack(triplanes, dim=0).to(device).reshape(-1, 3, 32, 64, 64)
        triplanes = triplanes.permute(1, 0, 2, 3, 4).reshape(3, -1, 32, 64, 64).contiguous()
    
        with torch.no_grad():
            coords_mc_one_time = coords_mc.clone().repeat(triplanes.shape[1], 1, 1).contiguous()
            mc_occup = auto_decoder(triplanes, coords_mc_one_time, sym=sym) # shape (batch_size, resolution^3, 1)
        
        mc_occup = mc_occup.squeeze(-1).detach().cpu().reshape(-1, resolution, resolution, resolution).numpy()
        mc_occup_ori =  mc_occup.copy()
        mc_occup_output = mc_occup.copy()
        # surface and inner = 1 outside = 0
        
        surface_and_inner_indices = np.where(mc_occup_output <= 0)
        outside_indices = np.where(mc_occup_output > 0)
        
        mc_occup_output[surface_and_inner_indices] = 1
        mc_occup_output[outside_indices] = 0
        
        mc_oocup_rm_outer = []
        valid_names = []
        ori_occups = []
        for i in range(mc_occup_output.shape[0]):
            struc = remove_outer_layers(mc_occup_output[i], (resolution, resolution, resolution))
            if max(struc.shape) > 0.85 * resolution:
                continue
            if min(struc.shape) < 0.75 * resolution:
                continue
            mc_oocup_rm_outer.append(struc)
            valid_names.append(batch_files[i])
            ori_occups.append(mc_occup_ori[i])
        
        # print("valid/total: ", len(mc_oocup_rm_outer), len(mc_occup_output))
        torch.cuda.empty_cache()
        if len(mc_oocup_rm_outer) == 0:
            continue

        E, nu, G, vf, CH, qualified, err = check_nu35(mc_oocup_rm_outer, batch_files)
        for i in range(len(valid_names)):
            now_name = valid_names[i]
            
            to_save_content = {
                'name' : now_name,
                'occup': ori_occups[i],
                'E': E[i],
                'nu': nu[i],
                'G': G[i],
                'CH': CH[i],
                'vf': vf[i],
                'err': err[i],
                'qualified': qualified[i],
            }
            
            np.save(f'{log_dir}/{now_name}.npy', to_save_content)
        torch.cuda.empty_cache()



def count_outer_layer_points(arr):
    """Count the points in the outermost layer of a 3D array."""
    
    # Count points on the front and back faces
    front_points = np.count_nonzero(arr[0, :, :])
    back_points = np.count_nonzero(arr[-1, :, :])
    
    # Count points on the left and right faces
    left_points = np.count_nonzero(arr[:, 0, :])
    right_points = np.count_nonzero(arr[:, -1, :])
    
    # Count points on the top and bottom faces
    top_points = np.count_nonzero(arr[:, :, 0])
    bottom_points = np.count_nonzero(arr[:, :, -1])
    
    # Total points in the outer layer
    total_outer_points = front_points + back_points + left_points + right_points + top_points + bottom_points
    
    return total_outer_points

def remove_outer_layer(arr):
    """Remove the outermost layer of a 3D array."""
    return arr[1:-1, 1:-1, 1:-1]  # Remove outer layer by slicing

def check_connective_mesh(v, f):
    """Check if the mesh is connected."""
    v = v.reshape(-1, 3)
    f = f.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    sepaerated = mesh.split()
    if len(sepaerated) > 1:
        return False
    return True  

def remove_outer_layers(arr, shape):
    arr = arr.reshape(shape)
    """Iteratively remove outer layers if their total points are less than or equal to 0.5*r*r."""
    r = arr.shape[0]
    
    while arr.shape[0] > 2:  # Ensure there are layers to remove
        threshold = 0.1 * r
        total_outer_points = count_outer_layer_points(arr)
        
        if total_outer_points <= threshold:
            arr = remove_outer_layer(arr)
            r = arr.shape[0]  # Update r as the array shrinks
        else:
            break  # Stop if the condition is not met
    
    return arr
    
def save_mesh(verts, faces, path):
    with open(path, 'w') as f:
        for v in verts:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for face in faces:
            f.write(f'f {" ".join([str(v+1) for v in face])}\n')


def setRandomSeed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()

