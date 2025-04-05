# %%

# from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from scheduler import linear_schedule, cosine_schedule
from forward import reverse_transform_pil, reverse_transform_tensor, sample_q, image_size, num_timesteps
from unet_model import DiffusionUnet
from reverse import sample_p, sampling
from torch.optim import Adam
from torchvision.transforms import GaussianBlur, ColorJitter
import tracemalloc
from custom_data_loader_celeba import get_celeba_dataloader

# Define the transformations
transform = Compose([
    Resize((image_size, image_size)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# %%
# Define noise functions
def add_gaussian_noise(image, mean=0, std=0.1):
    noise = torch.randn_like(image) * std + mean
    return image + noise

def add_speckle_noise(image, mean=0, std=0.1):
    noise = torch.randn_like(image) * std + mean
    return image + image * noise

def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = image.clone()
    salt_mask = torch.rand_like(image) < salt_prob
    pepper_mask = torch.rand_like(image) < pepper_prob
    noisy_image[salt_mask] = 1
    noisy_image[pepper_mask] = 0
    return noisy_image

def add_poisson_noise(image):
    noise = torch.poisson(image * 255) / 255
    return noise

def add_gaussian_blur(image, kernel_size=(5, 9), sigma=(0.1, 5.0)):
    blurrer = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blurrer(image)

def add_color_jitter(image, brightness=0.5, hue=0.3):
    jitter = ColorJitter(brightness=brightness, hue=hue)
    return jitter(image)


# %%
# Dataset and DataLoader
train_dataloader = get_celeba_dataloader(root='./data/celeba/img_align_celeba', batch_size=32, image_size=image_size)
test_dataloader = get_celeba_dataloader(root='./data/celeba/img_align_celeba', batch_size=32, image_size=image_size)

device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
class DDPM:
    def __init__(self, image_size=image_size, channels=3, dim_mults=(1, 2, 4, 8)):
        self.model = DiffusionUnet(dim=image_size, channels=channels, dim_mults=dim_mults).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=1e-4)
        self.device = device

    def train(self, dataloader, num_epochs, noise_type):
        self.model.train()
        results_folder = Path(f"./results_with_{noise_type}")
        results_folder.mkdir(exist_ok=True)
        start_time = time.time()
        tracemalloc.start()
        for epoch in range(num_epochs):
            for batch_index, batch in enumerate(dataloader):
                images = batch[0].to(self.device)  # Access images directly from the list
                t = torch.randint(1, num_timesteps, (images.shape[0],), device=self.device).long()
                noisy_images = self.apply_noise(images, noise_type)
                loss = self.compute_loss(noisy_images, t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_step = batch_index + epoch * len(dataloader)
                if current_step % 1000 == 0:
                    sample_images_list = sampling(self.model, (5, 3, image_size, image_size))
                    sample_images = torch.cat(sample_images_list, dim=0)
                    sample_images = reverse_transform_tensor(sample_images)
                    save_image(sample_images, str(results_folder / f'sample_{noise_type}_{current_step}.png'), nrow=5)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
            torch.save({
                'epoch': epoch + 1,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, f'/checkpoints/saved_model_with_{noise_type}_{epoch}.pth')
        end_time = time.time()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        training_time = end_time - start_time
        return training_time, peak_memory

    def compute_loss(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = sample_q(x0, t, noise)
        predicted_noise = self.model(x_t, t)
        loss = F.l1_loss(noise, predicted_noise)
        return loss

    def apply_noise(self, images, noise_type):
        if noise_type == 'gaussian':
            return add_gaussian_noise(images)
        elif noise_type == 'speckle':
            return add_speckle_noise(images)
        elif noise_type == 'salt_and_pepper':
            return add_salt_and_pepper_noise(images)
        elif noise_type == 'poisson':
            return add_poisson_noise(images)
        elif noise_type == 'gaussian_blur':
            return add_gaussian_blur(images)
        elif noise_type == 'color_jitter':
            return add_color_jitter(images)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)  # Access images directly from the list
                t = torch.randint(1, num_timesteps, (images.shape[0],), device=self.device).long()
                loss = self.compute_loss(images, t)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Average Loss: {avg_loss}")
        return avg_loss


# %%
# Training and evaluating DDPM with all noise types
noise_types = ['gaussian', 'speckle', 'salt_and_pepper', 'poisson', 'gaussian_blur', 'color_jitter']
best_model = None
best_model_name = None
best_avg_loss = float('inf')
performance_metrics = {}

for noise_type in noise_types:
    ddpm = DDPM()
    training_time, peak_memory = ddpm.train(train_dataloader, num_epochs=1000, noise_type=noise_type)
    avg_loss = ddpm.evaluate(test_dataloader)
    performance_metrics[noise_type] = {
        'avg_loss': avg_loss,
        'training_time': training_time,
        'peak_memory': peak_memory
    }
    if avg_loss < best_avg_loss:
        best_avg_loss = avg_loss
        best_model = ddpm
        best_model_name = noise_type


# %%
# Save the best model
torch.save({
    'model': best_model.model.state_dict(),
    'optimizer': best_model.optimizer.state_dict()
}, f'/checkpoints/best_model_{best_model_name}.pth')

print(f"Best model: {best_model_name} with average loss: {best_avg_loss}")


# %%
# Plot comparison of noise applied and denoised images
def plot_comparison(noise_type, ddpm):
    batch = next(iter(test_dataloader))
    images = batch[0].to(device)  # Access images directly from the list
    noisy_images = ddpm.apply_noise(images, noise_type)
    denoised_images = sampling(ddpm.model, (images.shape[0], 3, image_size, image_size))[-1]
    denoised_images = reverse_transform_tensor(denoised_images, noise_type)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i in range(5):
        axes[0, i].imshow(reverse_transform_pil(images[i]).permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        axes[1, i].imshow(reverse_transform_pil(noisy_images[i]).permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f"Noisy ({noise_type})")
        axes[1, i].axis('off')
        axes[2, i].imshow(reverse_transform_pil(denoised_images[i]).permute(1, 2, 0).cpu().numpy())
        axes[2, i].set_title("Denoised ({noise_type})")
        axes[2, i].axis('off')
    plt.show()


# %%
for noise_type in noise_types:
    ddpm = DDPM()
    ddpm.train(train_dataloader, num_epochs=1000, noise_type=noise_type)
    plot_comparison(noise_type, ddpm)


# %%
# Print performance metrics
for noise_type, metrics in performance_metrics.items():
    print(f"Noise type: {noise_type}")
    print(f"  Average Loss: {metrics['avg_loss']}")
    print(f"  Training Time: {metrics['training_time']} seconds")
    print(f"  Peak Memory Usage: {metrics['peak_memory']} bytes")
