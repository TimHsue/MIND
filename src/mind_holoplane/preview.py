"""Preview holoplane features, decoded SDF and mesh geometry."""
from pathlib import Path

import numpy as np


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _slice(axis, field, title, bound, figure, colorbar_axis=None):
    limit = max(float(np.abs(field).max()), 1e-12)
    plot = axis.imshow(field.T, origin='lower', extent=[-bound, bound, -bound, bound],
                       cmap='coolwarm', vmin=-limit, vmax=limit)
    axis.set_title(title)
    axis.set_xlabel('x'); axis.set_ylabel('z')
    if colorbar_axis is None:
        figure.colorbar(plot, ax=axis, shrink=0.8)
    else:
        figure.colorbar(plot, cax=colorbar_axis)


def save_geometry_preview(latent_path, field_path, path, title):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage.measure import marching_cubes
    plt = _plotting()
    latent = np.load(latent_path, allow_pickle=False)
    field = np.load(field_path, allow_pickle=False)
    if latent.ndim == 3:
        latent = latent.reshape(3, -1, *latent.shape[-2:])
    figure = plt.figure(figsize=(13, 4.3))
    axis = figure.add_axes([0.05, 0.18, 0.245, 0.62])
    magnitude = np.sqrt(np.mean(np.square(latent[0].astype(np.float64)), axis=0))
    plot = axis.imshow(magnitude.T, origin='lower', extent=[-1, 1, -1, 1], cmap='viridis')
    axis.set_title('Holoplane: YZ feature magnitude')
    axis.set_xlabel('y'); axis.set_ylabel('z')
    figure.colorbar(plot, cax=figure.add_axes([0.305, 0.18, 0.01, 0.62]))
    axis = figure.add_axes([0.39, 0.18, 0.245, 0.62])
    _slice(axis, field[:, field.shape[1]//2, :], 'Decoded SDF: central y slice', 1, figure,
           colorbar_axis=figure.add_axes([0.645, 0.18, 0.01, 0.62]))
    axis = figure.add_axes([0.715, 0.1, 0.26, 0.72], projection='3d')
    if float(field.min()) < 0 < float(field.max()):
        vertices, faces, _, _ = marching_cubes(field, level=0, spacing=tuple(2/(n-1) for n in field.shape))
        vertices -= 1
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)
        shade = 0.4 + 0.6*np.maximum(normals @ light, 0)
        colors = shade[:, None]*np.array([0.48, 0.68, 0.81])
        surface = Poly3DCollection(triangles, facecolors=colors, edgecolor='none', linewidth=0)
        axis.add_collection3d(surface)
    else:
        axis.text(0, 0, 0, 'No zero surface')
    axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel='x', ylabel='y', zlabel='z')
    axis.set_box_aspect((1, 1, 1), zoom=0.9)
    axis.set_axis_off()
    axis.view_init(elev=25, azim=-50)
    axis.set_title('OBJ surface: SDF = 0')
    figure.suptitle(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)
