from matplotlib import pyplot as plt
import numpy as np
import io
from PIL import Image


def visualize_sdf_clip(epoch, vis_id, occup_pred, writer, name=""):
    field = occup_pred[vis_id].detach().cpu().numpy()
    resolution = round(field.size ** (1 / 3))
    if resolution ** 3 != field.size:
        raise ValueError('SDF visualization requires a cubic grid')
    field = field.reshape((resolution,) * 3)
    slices = [min(resolution - 1, resolution * i // 4) for i in (1, 2, 3)]
    figure, axes = plt.subplots(3, 3, figsize=(12, 12))
    for column, index in enumerate(slices):
        for row, (plane, axis_name) in enumerate((
                (field[:, :, index], 'Z'), (field[:, index, :], 'Y'), (field[index, :, :], 'X'))):
            image = axes[row, column].imshow(plane.T, origin='lower', cmap='viridis')
            axes[row, column].set_title(f'{axis_name} index = {index}')
            figure.colorbar(image, ax=axes[row, column])
    figure.tight_layout()
    with io.BytesIO() as buffer:
        figure.savefig(buffer, format='png')
        buffer.seek(0)
        image = np.array(Image.open(buffer))
    writer.add_image(f'SDFClip/epoch_{epoch}_{name}', image, epoch, dataformats='HWC')
    plt.close(figure)


def visualize_holoplane(epoch, vis_id, holoplane_latents, writer):
    plane_x = holoplane_latents[0][vis_id].to('cpu').detach().numpy() # shape c, r, r
    plane_y = holoplane_latents[1][vis_id].to('cpu').detach().numpy()
    plane_z = holoplane_latents[2][vis_id].to('cpu').detach().numpy()

    planes = np.stack([plane_x, plane_y, plane_z], axis=0) # shape 3, c, r, r

    mean_planes = np.mean(planes, axis=1)
    max_planes = np.max(planes, axis=1)
    min_planes = np.min(planes, axis=1)

    # Set up the figure for visualization
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Titles for the subplots
    titles = ['Mean', 'Max', 'Min']

    # Plotting the mean, max, and min for each holoplane
    for i, (mean_plane, max_plane, min_plane) in enumerate(zip(mean_planes, max_planes, min_planes)):
        im = axes[i, 0].imshow(mean_plane, cmap='viridis')
        axes[i, 0].set_title(f'Holoplane {i+1} - {titles[0]}')
        fig.colorbar(im, ax=axes[i, 0])

        im = axes[i, 1].imshow(max_plane, cmap='viridis')
        axes[i, 1].set_title(f'Holoplane {i+1} - {titles[1]}')
        fig.colorbar(im, ax=axes[i, 1])

        im = axes[i, 2].imshow(min_plane, cmap='viridis')
        axes[i, 2].set_title(f'Holoplane {i+1} - {titles[2]}')
        fig.colorbar(im, ax=axes[i, 2])

    plt.tight_layout()
    # Convert the figure to a NumPy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(Image.open(buf))

    writer.add_image(f'Holoplanes/epoch_{epoch}', img, epoch, dataformats='HWC')

    plt.clf()
    plt.close()


def visualize_occupancy(epoch, vis_id, coordinates, occup_pred, occup_gd, writer, name='SDF'):
    fig = plt.figure(figsize=(20, 20))

    to_vis_coords = coordinates[vis_id].to('cpu').detach().numpy()
    to_vis_occup = occup_pred[vis_id].to('cpu').detach().numpy()
    to_vis_occup_gd = occup_gd[vis_id].unsqueeze(-1).to('cpu').detach().numpy()

    sdf = np.concatenate((to_vis_coords, to_vis_occup), axis=1)
    sdf_lt_0_01 = sdf[(sdf[:, 3] < 0.01)]

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

    # 误差归一化，使其范围在 [0, 0.01]
    sdf_value_delta_clipped = np.clip(sdf_value_delta, -0.05, 0.05)
    sdf_value_normalized = sdf_value_delta_clipped / 0.05

    # 根据误差值设置透明度，误差为0时透明，误差为0.01时完全不透明
    alpha_values = np.abs(sdf_value_normalized.squeeze())

    # 设置颜色：误差小于0的点为蓝色，误差大于0的点为红色
    num_points = to_vis_coords.shape[0]
    colors = np.zeros((num_points, 4))  # RGBA (num_points, 4)

    # 蓝色（误差小）
    mask_blue = sdf_value_normalized < 0
    mask_blue = mask_blue.squeeze()
    colors[mask_blue, 2] = 1  # 蓝色通道
    colors[mask_blue, 3] = alpha_values[mask_blue]  # 透明度

    # 红色（误差大）
    mask_red = sdf_value_normalized > 0
    mask_red = mask_red.squeeze()
    colors[mask_red, 0] = 1  # 红色通道
    colors[mask_red, 3] = alpha_values[mask_red]  # 透明度

    # 创建3D图
    ax1 = fig.add_subplot(111, projection='3d')
    sc1 = ax1.scatter(to_vis_coords[:, 0], to_vis_coords[:, 1], to_vis_coords[:, 2], c=colors, s=1)

    ax1.set_title('PR SDF Error')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    plt.tight_layout()

    # Convert the figure to a NumPy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = np.array(Image.open(buf))

    writer.add_image(f'SDFE/epoch_{epoch}', img, epoch, dataformats='HWC')

    plt.clf()
    plt.close()
