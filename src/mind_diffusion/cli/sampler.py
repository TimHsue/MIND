# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

# Conditional Heun sampling for individual microstructure cells.
import numpy as np
import torch

def heun_sampler(
    diffusion_model, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, cond_strength=1.0,
    drop_labels=None
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, diffusion_model.sigma_min)
    sigma_max = min(sigma_max, diffusion_model.sigma_max)
    drop_mask = torch.tensor(0, dtype=class_labels.dtype, device=class_labels.device)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([diffusion_model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    x_next = latents.to(torch.float32) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:]), start=1):
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = diffusion_model.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        t_hat_input = t_hat.unsqueeze(0).repeat(x_hat.shape[0], 1)

        with torch.no_grad():
            denoised = diffusion_model(x_hat, t_hat_input, class_labels.clone(), drop_mask=drop_labels)
            denoised_drop = diffusion_model(x_hat, t_hat_input, class_labels.clone(), drop_mask=drop_mask)
        denoised = denoised * cond_strength + denoised_drop * (1 - cond_strength)

        d_cur = (x_hat - denoised) / t_hat

        x_next = x_hat + (t_next - t_hat) * d_cur
        x_next = x_next.detach()

        # Apply Heun correction; the last two intervals use Euler updates.
        if i < num_steps - 1:
            t_next_input = t_next.unsqueeze(0).repeat(x_hat.shape[0], 1)
            with torch.no_grad():
                denoised = diffusion_model(x_next, t_next_input, class_labels.clone(), drop_mask=drop_labels)
                denoised_drop = diffusion_model(x_next, t_next_input, class_labels.clone(), drop_mask=drop_mask)
            denoised = denoised * cond_strength + denoised_drop * (1 - cond_strength)


            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)


    return x_next.detach()
