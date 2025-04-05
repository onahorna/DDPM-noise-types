# salt-and-pepper noise diffusion model
# This code implements the reverse process for a diffusion model that handles salt-and-pepper noise.
import torch
import torch.nn as nn
import torch.nn.functional as F
from forward import sample_by_t
from scheduler import linear_schedule

device = "cuda" if torch.cuda.is_available() else "cpu"

# Precalculate the values of betas and alphas
num_timesteps = 1000
betas_t = linear_schedule(num_timesteps)

alphas_t = 1. - betas_t
alphas_bar_t = torch.cumprod(alphas_t, dim=0)
alphas_bar_t_minus_1 = torch.cat((torch.tensor([0]), alphas_bar_t[:-1]))
one_over_sqrt_alphas_t = 1. / torch.sqrt(alphas_t)
sqrt_alphas_t = torch.sqrt(alphas_t)
sqrt_alphas_bar_t = torch.sqrt(alphas_bar_t)
sqrt_alphas_bar_t_minus_1 = torch.sqrt(alphas_bar_t_minus_1)
sqrt_1_minus_alphas_bar_t = torch.sqrt(1. - alphas_bar_t)

# The variance of q(xₜ₋₁ | xₜ, x₀) as in part 3
posterior_variance = (1. - alphas_bar_t_minus_1) / (1. - alphas_bar_t) * betas_t

# Reverse Process Sampling
@torch.no_grad()
def sample_p(model, x_t, t, clipping=True):
    """
    Sample from p_θ(xₜ₋₁|xₜ) to get xₜ₋₁ according to Algorithm 2.
    This version is adapted for salt-and-pepper noise.
    """
    betas_t_sampled = sample_by_t(betas_t, t, x_t.shape)
    sqrt_1_minus_alphas_bar_t_sampled = sample_by_t(sqrt_1_minus_alphas_bar_t, t, x_t.shape)
    one_over_sqrt_alphas_t_sampled = sample_by_t(one_over_sqrt_alphas_t, t, x_t.shape)

    if clipping:
        sqrt_alphas_bar_t_sampled = sample_by_t(sqrt_alphas_bar_t, t, x_t.shape)
        sqrt_alphas_bar_t_minus_1_sampled = sample_by_t(sqrt_alphas_bar_t_minus_1, t, x_t.shape)
        alphas_bar_t_sampled = sample_by_t(alphas_bar_t, t, x_t.shape)
        sqrt_alphas_t_sampled = sample_by_t(sqrt_alphas_t, t, x_t.shape)
        alphas_bar_t_minus_1_sampled = sample_by_t(alphas_bar_t_minus_1, t, x_t.shape)

        # Denoising for salt-and-pepper noise (this assumes the model can handle salt-and-pepper noise)
        x0_reconstruct = 1 / sqrt_alphas_bar_t_sampled * (x_t - sqrt_1_minus_alphas_bar_t_sampled * model(x_t, t))
        x0_reconstruct = torch.clip(x0_reconstruct, -1., 1.)

        predicted_mean = (sqrt_alphas_bar_t_minus_1_sampled * betas_t_sampled) / (1 - alphas_bar_t_sampled) * x0_reconstruct + \
                          (sqrt_alphas_t_sampled * (1 - alphas_bar_t_minus_1_sampled)) / (1 - alphas_bar_t_sampled) * x_t
    else:
        # When no clipping is performed, use the following:
        predicted_mean = one_over_sqrt_alphas_t_sampled * (x_t - betas_t_sampled / sqrt_1_minus_alphas_bar_t_sampled * model(x_t, t))

    # If we are at the last timestep, return the predicted mean without adding noise
    if t[0].item() == 1:
        return predicted_mean
    else:
        posterior_variance_sampled = sample_by_t(posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)  # Adding noise to the predicted mean
        return predicted_mean + torch.sqrt(posterior_variance_sampled) * noise

from tqdm import tqdm

@torch.no_grad()
def sampling(model, shape, image_noise_steps_to_keep=1):
    """
    Implementing Algorithm 2 - sampling.
    Args:
      model (torch.Module): the model that predicts the noise.
      shape (tuple): shape of the data (batch, channels, image_size, image_size)
    Returns:
      (list): list containing the images in the different steps of the reverse process.
    """
    batch = shape[0]
    images = torch.randn(shape, device=device)  # pure noise
    images_list = []

    for timestep in tqdm(range(num_timesteps, 0, -1), desc='sampling timestep'):
        images = sample_p(model, images, torch.full((batch,), timestep, device=device, dtype=torch.long))
        if timestep <= image_noise_steps_to_keep:
            images_list.append(images.cpu())

    return images_list
