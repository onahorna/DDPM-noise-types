# Gaussian Blur Noise

# %%
# WORKS FOR One image

import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import GaussianBlur, Compose, ToTensor, CenterCrop, Resize, Normalize
from scheduler import linear_schedule
import torch
from torchvision.transforms import ToPILImage
from custom_data_loader_celeba import get_celeba_dataloader

# Define the transformations
image_size = 64

# Directly load the image (from the specific path)
image_path = '.\\data\\celeba\\img_align_celeba\\img_align_celeba\\000002.jpg' #'/data/celeba/img_align_celeba/img_align_celeba/000002.jpg'
image = Image.open(image_path).convert('RGB')

# Preprocess the image
transform = Compose([
    Resize((image_size, image_size)),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Apply the transformation
x0 = transform(image).unsqueeze(0)  # Add batch dimension

# Reverse transform for visualizing
reverse_transform_pil = Compose([
    Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
    ToPILImage()
])

reverse_transform_tensor = Compose([
    Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
])

# Create the output directory if it doesn't exist
output_folder = 'output_all_noises'
os.makedirs(output_folder, exist_ok=True)

def sample_by_t(tensor_to_sample, timesteps, x_shape):
    batch_size = timesteps.shape[0]
    sampled_tensor = tensor_to_sample.gather(-1, timesteps.cpu())
    sampled_tensor = torch.reshape(sampled_tensor, (batch_size,) + (1,) * (len(x_shape) - 1))
    return sampled_tensor.to(timesteps.device)

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

# the variance of q(xₜ₋₁ | xₜ, x₀) as in part 3
posterior_variance = (1. - alphas_bar_t_minus_1) / (1. - alphas_bar_t) * betas_t

def sample_q(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_alphas_bar_t_sampled = sample_by_t(sqrt_alphas_bar_t, t, x0.shape)
    sqrt_1_minus_alphas_bar_t_sampled = sample_by_t(sqrt_1_minus_alphas_bar_t, t, x0.shape)
    x_t = sqrt_alphas_bar_t_sampled * x0 + sqrt_1_minus_alphas_bar_t_sampled * noise
    return x_t

def get_noisy_image(x0, t, transform=reverse_transform_pil):
    x_noisy = sample_q(x0, t)
    # Apply GaussianBlur noise instead of Gaussian noise
    gaussian_blurred_image = GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))(reverse_transform_pil(x_noisy.squeeze()))
    return gaussian_blurred_image

def save_noisy_images(noisy_images, noise_type, original_image_name):
    for i, image in enumerate(noisy_images):
        image_name = f"forward_{noise_type}_{original_image_name}_{i}.png"
        image_path = os.path.join(output_folder, image_name)
        image.save(image_path)
        print(f"Saved: {image_path}")

def show_noisy_images(noisy_images, noise_type, original_image_name):
    num_of_image_sets = len(noisy_images)
    num_of_images_in_set = len(noisy_images[0])
    image_size = noisy_images[0][0].size[0]

    full_image = Image.new('RGB', (image_size * num_of_images_in_set + (num_of_images_in_set - 1), image_size * num_of_image_sets + (num_of_image_sets - 1)))
    for set_index, image_set in enumerate(noisy_images):
        for image_index, image in enumerate(image_set):
            full_image.paste(image, (image_index * image_size + image_index, set_index * image_size + set_index))

    # Save the combined noisy images
    save_noisy_images([full_image], noise_type, original_image_name)
    plt.imshow(full_image)
    plt.axis('off')
    return full_image

t = torch.tensor([20])

# Generate and save noisy images
noisy_images = [[get_noisy_image(x0, torch.tensor([t])) for t in [0, 50, 100, 150, 200]]]
show_noisy_images(noisy_images, noise_type='gaussian_blur', original_image_name=os.path.basename(image_path))
# %%