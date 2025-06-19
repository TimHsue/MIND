import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import io
from PIL import Image

def existsOrMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

def fig_to_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)


def visualize_sdf_norm_clip(epoch, vis_id, occup_pred, writer):
    fig = plt.figure(figsize=(12, 6))

    to_vis_occup = occup_pred[vis_id].to('cpu').detach().numpy()# 128 ^ 3, 1
    to_vis_occup = to_vis_occup.reshape((128, 128, 128))
    
    error_map = np.clip(to_vis_occup, 0, 5)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=1, vmax=5)
    cmap = plt.get_cmap('coolwarm')  # 使用 coolwarm 色图，效果更加直观
    # vis clip plane: x, y, z = 32, 64, 96
    
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    
    for i in range(0, 9):
        if i < 3:
            im = ax[0, i].imshow(error_map[:, :, 32 + i * 32],  cmap=cmap, norm=norm)
            ax[0, i].set_title(f'Z = {i * 32}')
            fig.colorbar(im, ax=ax[0, i])
        if i >= 3 and i < 6:
            im = ax[1, i-3].imshow(error_map[:, 32 + (i - 3) * 32, :],  cmap=cmap, norm=norm)
            ax[1, i-3].set_title(f'Y = {i * 32}')
            fig.colorbar(im, ax=ax[1, i-3])
        if i >= 6:
            im = ax[2, i-6].imshow(error_map[32 + (i - 6) * 32, :, :],  cmap=cmap, norm=norm)
            ax[2, i-6].set_title(f'X = {i * 32}')
            fig.colorbar(im, ax=ax[2, i-6])
            
    plt.tight_layout()
    
    # Convert the figure to a NumPy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    writer.add_image(f'SDFNorm/epoch_{epoch}', img, epoch, dataformats='HWC')
    fig.clf()
    fig.clear()
    plt.clf()
    plt.close()
    
    

def visualize_sdf_clip(epoch, vis_id, occup_pred, writer, name=""):
    fig = plt.figure(figsize=(12, 6))

    to_vis_occup = occup_pred[vis_id].to('cpu').detach().numpy()# 128 ^ 3, 1
    to_vis_occup = to_vis_occup.reshape((128, 128, 128))
    
    # vis clip plane: x, y, z = 32, 64, 96
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))
    
    
    for i in range(0, 9):
        if i < 3:
            im = ax[0, i].imshow(to_vis_occup[:, :, 32 + i * 32], cmap='viridis')
            ax[0, i].set_title(f'Z = {i * 32}')
            fig.colorbar(im, ax=ax[0, i])
        if i >= 3 and i < 6:
            im = ax[1, i-3].imshow(to_vis_occup[:, 32 + (i - 3) * 32, :], cmap='viridis')
            ax[1, i-3].set_title(f'Y = {i * 32}')
            fig.colorbar(im, ax=ax[1, i-3])
        if i >= 6:
            im = ax[2, i-6].imshow(to_vis_occup[32 + (i - 6) * 32, :, :], cmap='viridis')
            ax[2, i-6].set_title(f'X = {i * 32}')
            fig.colorbar(im, ax=ax[2, i-6])
            
    plt.tight_layout()
    
    # Convert the figure to a NumPy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    writer.add_image(f'SDFClip/epoch_{epoch}_{name}', img, epoch, dataformats='HWC')
    plt.clf()
    plt.close()

def visualize_triplane(epoch, vis_id, triplane_latents, writer):
    plane_x = triplane_latents[0][vis_id].to('cpu').detach().numpy() # shape c, r, r
    plane_y = triplane_latents[1][vis_id].to('cpu').detach().numpy()
    plane_z = triplane_latents[2][vis_id].to('cpu').detach().numpy()
    
    planes = np.stack([plane_x, plane_y, plane_z], axis=0) # shape 3, c, r, r
    
    mean_planes = np.mean(planes, axis=1)
    max_planes = np.max(planes, axis=1)
    min_planes = np.min(planes, axis=1)

    # Set up the figure for visualization
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Titles for the subplots
    titles = ['Mean', 'Max', 'Min']

    # Plotting the mean, max, and min for each triplane
    for i, (mean_plane, max_plane, min_plane) in enumerate(zip(mean_planes, max_planes, min_planes)):
        im = axes[i, 0].imshow(mean_plane, cmap='viridis')
        axes[i, 0].set_title(f'Triplane {i+1} - {titles[0]}')
        fig.colorbar(im, ax=axes[i, 0])

        im = axes[i, 1].imshow(max_plane, cmap='viridis')
        axes[i, 1].set_title(f'Triplane {i+1} - {titles[1]}')
        fig.colorbar(im, ax=axes[i, 1])

        im = axes[i, 2].imshow(min_plane, cmap='viridis')
        axes[i, 2].set_title(f'Triplane {i+1} - {titles[2]}')
        fig.colorbar(im, ax=axes[i, 2])
        
    plt.tight_layout()
    # Convert the figure to a NumPy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    writer.add_image(f'Triplanes/epoch_{epoch}', img, epoch, dataformats='HWC')

    plt.clf()
    plt.close()

def visualize_voxel(epoch, vis_id, coordinates, occup_pred, writer):
    fig = plt.figure(figsize=(12, 6))

    to_vis_coords = coordinates[vis_id].to('cpu').detach().numpy()
    to_vis_occup = occup_pred[vis_id].to('cpu').detach().numpy()
    
    sdf = np.concatenate((to_vis_coords, to_vis_occup), axis=1)
    sdf_lt_0_01 = sdf[(sdf[:, 3] < 0.01)]
    
    
    # Subplot 1: All SDF values
    ax1 = fig.add_subplot(121, projection='3d')
    sc1 = ax1.scatter(sdf[:, 0], sdf[:, 1], sdf[:, 2], c=sdf[:, 3], cmap='viridis', s=1)
    plt.colorbar(sc1, ax=ax1, label='SDF Value')
    ax1.set_title('PR All SDF Values')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Subplot 2: SDF < 0.01 (includes SDF < 0.001 and SDF < 0.01)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(sdf_lt_0_01[:, 0], sdf_lt_0_01[:, 1], sdf_lt_0_01[:, 2], color='green', s=1)
    ax2.set_title('PR SDF < 0.01')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    
    # Convert the figure to a NumPy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    writer.add_image(f'SDFMC/epoch_{epoch}', img, epoch, dataformats='HWC')

    plt.clf()
    plt.close()
 
def visualize_occupancy(epoch, vis_id, coordinates, occup_pred, occup_gd, writer, name='SDF'):
    fig = plt.figure(figsize=(20, 20))

    to_vis_coords = coordinates[vis_id].to('cpu').detach().numpy()
    to_vis_occup = occup_pred[vis_id].to('cpu').detach().numpy()
    to_vis_occup_gd = occup_gd[vis_id].unsqueeze(-1).to('cpu').detach().numpy()
    
    sdf = np.concatenate((to_vis_coords, to_vis_occup), axis=1)
    sdf_lt_0_01 = sdf[(sdf[:, 3] < 0.01)]
    
    # print(to_vis_coords.shape, to_vis_occup.shape, to_vis_occup_gd.shape)
    sdf_gd = np.concatenate((to_vis_coords, to_vis_occup_gd), axis=1)
    sdf_lt_0_01_gd = sdf_gd[(sdf_gd[:, 3] < 0.01)]
    
    # subplot 3: All SDF values from ground truth
    ax3 = fig.add_subplot(221, projection='3d')
    sc3 = ax3.scatter(sdf_gd[:, 0], sdf_gd[:, 1], sdf_gd[:, 2], c=sdf_gd[:, 3], cmap='viridis', s=1)
    plt.colorbar(sc3, ax=ax3, label='SDF Value')
    ax3.set_title('GT All SDF Values')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    
    # Subplot 1: All SDF values
    ax1 = fig.add_subplot(222, projection='3d')
    sc1 = ax1.scatter(sdf[:, 0], sdf[:, 1], sdf[:, 2], c=sdf[:, 3], cmap='viridis', s=1)
    plt.colorbar(sc1, ax=ax1, label='SDF Value')
    ax1.set_title('PR All SDF Values')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Subplot 4: SDF < 0.01 (includes SDF < 0.001 and SDF < 0.01)
    ax4 = fig.add_subplot(223, projection='3d')
    ax4.scatter(sdf_lt_0_01_gd[:, 0], sdf_lt_0_01_gd[:, 1], sdf_lt_0_01_gd[:, 2], color='green', s=1)
    ax4.set_title('GT SDF < 0.01')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    # Subplot 2: SDF < 0.01 (includes SDF < 0.001 and SDF < 0.01)

    ax2 = fig.add_subplot(224, projection='3d')
    ax2.scatter(sdf_lt_0_01[:, 0], sdf_lt_0_01[:, 1], sdf_lt_0_01[:, 2], color='green', s=1)
    ax2.set_title('PR SDF < 0.01')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    plt.tight_layout()
    
    # Convert the figure to a NumPy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    writer.add_image(f'{name}/epoch_{epoch}', img, epoch, dataformats='HWC')

    plt.clf()
    plt.close()

def visualize_occupancy_error(epoch, vis_id, coordinates, occup_pred, occup_gd, writer):
    fig = plt.figure(figsize=(12, 12))

    to_vis_coords = coordinates[vis_id].to('cpu').detach().numpy()
    to_vis_occup = occup_pred[vis_id].to('cpu').detach().numpy()
    to_vis_occup_gd = occup_gd[vis_id].unsqueeze(-1).to('cpu').detach().numpy()
    
    sdf_value_delta = to_vis_occup - to_vis_occup_gd
    
    sdf_value_delta_clipped = np.clip(sdf_value_delta, -0.05, 0.05)
    sdf_value_normalized = sdf_value_delta_clipped / 0.05
    
    alpha_values = np.abs(sdf_value_normalized.squeeze())
    
    num_points = to_vis_coords.shape[0]
    colors = np.zeros((num_points, 4))  
    
    mask_blue = sdf_value_normalized < 0
    mask_blue = mask_blue.squeeze()
    colors[mask_blue, 2] = 1 
    colors[mask_blue, 3] = alpha_values[mask_blue] 

    mask_red = sdf_value_normalized > 0
    mask_red = mask_red.squeeze()
    colors[mask_red, 0] = 1 
    colors[mask_red, 3] = alpha_values[mask_red] 
    

    ax1 = fig.add_subplot(111, projection='3d')
    sc1 = ax1.scatter(to_vis_coords[:, 0], to_vis_coords[:, 1], to_vis_coords[:, 2], c=colors, s=1)
    
    ax1.set_title('PR SDF Error')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    writer.add_image(f'SDFE/epoch_{epoch}', img, epoch, dataformats='HWC')

    plt.clf()
    plt.close()

def visualize_mesh(epoch, verts, faces, writer, name="MeshPR"):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], lw=1)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(Image.open(buf))
    
    writer.add_image(f'{name}/epoch_{epoch}', img, epoch, dataformats='HWC')

    plt.clf()
    plt.close()
