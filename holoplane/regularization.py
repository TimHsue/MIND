

import torch


def edr_loss(batch_size, triplane_latents, auto_decoder, device='cuda', offset_distance=0.01):

    num_points = 10000
    random_coords = torch.rand(batch_size, num_points, 3).to(device) * 2 - 1 # sample from [-1, 1]
    offset_coords = random_coords + torch.randn_like(random_coords) * offset_distance # Make offset_magnitude bigger if you want smoother
    densities_initial = auto_decoder(triplane_latents, random_coords)
    densities_offset = auto_decoder(triplane_latents, offset_coords)
    delta = densities_offset - densities_initial
    density_smoothness_loss = (delta ** 2).sum() / batch_size

    return density_smoothness_loss

'''
def sym_loss_fun(triplane_latents):
    sym_loss = 0
    for triplane in triplane_latents:
        sym_loss += (triplane - triplane.flip(2)).pow(2).sum() / triplane.shape[0]
        sym_loss += (triplane - triplane.flip(3)).pow(2).sum() / triplane.shape[0]
        sym_loss += (triplane - triplane.permute(0, 1, 3, 2)).pow(2).sum() / triplane.shape[0]
    
    return sym_loss
'''

def sym_loss_fun(triplane_latents):
    sym_loss = 0
    for triplane in triplane_latents:
        rotated = torch.rot90(triplane, 1, [2, 3])
        flipped = triplane.flip(dims=[3])

        loss_rot = (triplane - rotated).pow(2).sum() / triplane.shape[0]
        loss_flip = (triplane - flipped).pow(2).sum() / triplane.shape[0]

        sym_loss += loss_rot + loss_flip
    return sym_loss

def l2_reg(triplane_latents):
    l2_loss = 0
    
    for triplane in triplane_latents:
        # Compute L2 loss as sum of squares of all elements
        l2_loss += triplane.pow(2).mean()
    
    return l2_loss / 3.0

def tv_reg(triplane_latents):
    tv_loss = 0
    for triplane in triplane_latents:
        # Compute differences in height and width directions
        tv_h = (triplane[:, :, 1:, :] - triplane[:, :, :-1, :]).pow(2).mean()
        tv_w = (triplane[:, :, :, 1:] - triplane[:, :, :, :-1]).pow(2).mean()
        tv_loss += (tv_h + tv_w)
    
    return tv_loss / 3.0 