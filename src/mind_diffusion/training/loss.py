# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch

from mind_diffusion.torch_utils import persistence

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
