import torch


def edr_loss(batch_size, holoplane_latents, auto_decoder, device='cuda', offset_distance=0.01):

    num_points = 10000
    random_coords = torch.rand(batch_size, num_points, 3).to(device) * 2 - 1 # sample from [-1, 1]
    offset_coords = random_coords + torch.randn_like(random_coords) * offset_distance # Make offset_magnitude bigger if you want smoother
    densities_initial = auto_decoder(holoplane_latents, random_coords)
    densities_offset = auto_decoder(holoplane_latents, offset_coords)
    delta = densities_offset - densities_initial
    density_smoothness_loss = (delta ** 2).sum() / batch_size

    return density_smoothness_loss


def sym_loss_fun(holoplane_latents):
    sym_loss = 0
    for holoplane in holoplane_latents:
        rotated = torch.rot90(holoplane, 1, [2, 3])
        flipped = holoplane.flip(dims=[3])

        loss_rot = (holoplane - rotated).pow(2).sum() / holoplane.shape[0]
        loss_flip = (holoplane - flipped).pow(2).sum() / holoplane.shape[0]

        sym_loss += loss_rot + loss_flip
    return sym_loss

def l2_reg(holoplane_latents):
    l2_loss = 0

    for holoplane in holoplane_latents:
        # Compute L2 loss as sum of squares of all elements
        l2_loss += holoplane.pow(2).mean()

    return l2_loss / 3.0

def tv_reg(holoplane_latents):
    tv_loss = 0
    for holoplane in holoplane_latents:
        # Compute differences in height and width directions
        tv_h = (holoplane[:, :, 1:, :] - holoplane[:, :, :-1, :]).pow(2).mean()
        tv_w = (holoplane[:, :, :, 1:] - holoplane[:, :, :, :-1]).pow(2).mean()
        tv_loss += (tv_h + tv_w)

    return tv_loss / 3.0
