# %%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize
from torchvision.utils import save_image, make_grid
from pathlib import Path
import optuna
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
# from datasets import load_dataset
# from torchvision.transforms import v2
from torchvision.transforms import GaussianBlur, ColorJitter
from custom_data_loader_celeba import get_celeba_dataloader
from unet_model import DiffusionUnet
from forward import sample_q, reverse_transform_pil, reverse_transform_tensor
from reverse import sample_p, sampling
from scheduler import linear_schedule, cosine_schedule

# %%
# Save study results
results_folder = Path("./optuna_results")
results_folder.mkdir(exist_ok=True)

# Define constants
image_size = 64
num_timesteps = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define noise types
def add_noise(image, noise_type):
    if noise_type == "gaussian":
        noise = torch.randn_like(image)
        return image + noise
    elif noise_type == "speckle":
        noise = torch.randn_like(image) * image
        return image + noise
    elif noise_type == "salt_and_pepper":
        prob = 0.05
        noisy = image.clone()
        salt = torch.rand_like(image) < prob / 2
        pepper = torch.rand_like(image) < prob / 2
        noisy[salt] = 1
        noisy[pepper] = 0
        return noisy
    elif noise_type == "poisson":
        image = (image + 1) / 2  # Scale to [0, 1]
        noise = torch.poisson(image * 255) / 255
        noise = noise * 2 - 1  # Scale back to [-1, 1]
        return image + noise
    elif noise_type == "gaussian_blur":
        return add_gaussian_blur(image)
    elif noise_type == "color_jitter":
        return add_color_jitter(image)
    else:
        raise ValueError("Unsupported noise type")

def add_gaussian_blur(image, kernel_size=(5, 9), sigma=(0.1, 5.0)):
    blurrer = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    return blurrer(image)

def add_color_jitter(image, brightness=0.5, hue=0.3):
    jitter = ColorJitter(brightness=brightness, hue=hue)
    return jitter(image)

# %%
# Define dataset and dataloader
train_transform = Compose([
    Resize((image_size, image_size)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_transform = Compose([
    Resize((image_size, image_size)),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# celeba = load_dataset("data/celeba", split="train")
# train_dataset = celeba.map(lambda x: {"image": train_transform(x["image"])})
# test_dataset = celeba.map(lambda x: {"image": test_transform(x["image"])})

# batch_size = 32
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_dataloader = get_celeba_dataloader(root='./data/celeba/img_align_celeba', batch_size=32, image_size=image_size)
test_dataloader = get_celeba_dataloader(root='./data/celeba/img_align_celeba', batch_size=32, image_size=image_size)

# %%
# Define loss computation
def compute_loss(model, x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    x_t = sample_q(x0, t, noise)
    predicted_noise = model(x_t, t)
    loss = F.l1_loss(noise, predicted_noise)
    return loss

# Define evaluation function
def evaluate(model, dataloader, device, noise_type):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            images = add_noise(images, noise_type)
            t = torch.randint(1, num_timesteps, (images.shape[0],), device=device).long()
            loss = compute_loss(model, images, t)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Define Optuna objective function
def objective(trial):
    # Hyperparameters to tune
    dim = trial.suggest_int("dim", 32, 128)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    noise_type = trial.suggest_categorical("noise_type", ["gaussian", "speckle", "salt_and_pepper", "poisson", "gaussian_blur", "color_jitter"])

    # Model and optimizer
    model = DiffusionUnet(dim=image_size, channels=3, dim_mults=(1, 2, 4, 8)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            images = batch[0].to(device)
            images = add_noise(images, noise_type)
            t = torch.randint(1, num_timesteps, (images.shape[0],), device=device).long()
            loss = compute_loss(model, images, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    test_loss = evaluate(model, test_dataloader, device, noise_type)

    # Save the model
    torch.save(model.state_dict(), results_folder / f"model_{noise_type}_{trial.number}.pth")
    return test_loss

# Run Optuna optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Save study results
results_folder = Path("./optuna_results")
results_folder.mkdir(exist_ok=True)
study.trials_dataframe().to_csv(results_folder / "study_results.csv")

# Plot results
def plot_results(study):
    df = study.trials_dataframe()
    fig, ax = plt.subplots()
    for noise_type in df["params_noise_type"].unique():
        subset = df[df["params_noise_type"] == noise_type]
        ax.plot(subset["number"], subset["value"], label=noise_type)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Test Loss")
    ax.legend()
    plt.savefig(results_folder / "results_plot.png")

plot_results(study)

# Compare best models with ground truth
def compare_best_models(study):
    best_trials = study.best_trials
    fig, axes = plt.subplots(len(best_trials), 2, figsize=(10, 5 * len(best_trials)))
    for i, trial in enumerate(best_trials):
        model = DiffusionUnet(dim=image_size, channels=3, dim_mults=(1, 2, 4, 8)).to(device)
        model.load_state_dict(torch.load(results_folder / f"model_{trial.params['noise_type']}_{trial.number}.pth"))
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                images = batch[0].to(device)
                noisy_images = add_noise(images, trial.params["noise_type"])
                sampled_images = sampling(model, noisy_images.shape)
                axes[i, 0].imshow(make_grid(reverse_transform_tensor(noisy_images)).permute(1, 2, 0).cpu().numpy())
                axes[i, 1].imshow(make_grid(reverse_transform_tensor(sampled_images)).permute(1, 2, 0).cpu().numpy())
                break
    plt.savefig(results_folder / "comparison_plot.png")

compare_best_models(study)
