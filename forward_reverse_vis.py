import torch
import matplotlib.pyplot as plt
from Diffusion_combined import DDPM, add_gaussian_noise, add_speckle_noise, add_salt_and_pepper_noise, add_poisson_noise, add_gaussian_blur, add_color_jitter
from custom_data_loader_celeba import get_celeba_dataloader
import os

# Load CelebA dataset using the custom data loader
dataloader = get_celeba_dataloader(root='./data/celeba/img_align_celeba', batch_size=32, image_size=64)

def train_model(ddpm, dataloader, optimizer, num_epochs, noise_type, model_id):
    ddpm.train()
    best_loss = float('inf')
    checkpoint_dir = f'checkpoints_diffusion/model_{noise_type}_{model_id}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            data = batch[0].to(ddpm.device)

            if noise_type == 'gaussian':
                data = add_gaussian_noise(data)
            elif noise_type == 'speckle':
                data = add_speckle_noise(data)
            elif noise_type == 'salt_and_pepper':
                data = add_salt_and_pepper_noise(data)
            elif noise_type == 'poisson':
                data = add_poisson_noise(data)
            elif noise_type == 'gaussian_blur':
                data = add_gaussian_blur(data)
            elif noise_type == 'color_jitter':
                data = add_color_jitter(data)

            optimizer.zero_grad()
            loss = ddpm.compute_loss(data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ddpm.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))

def evaluate_model(ddpm, dataloader, noise_type):
    ddpm.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            data = batch[0].to(ddpm.device)
            if noise_type == 'gaussian':
                data = add_gaussian_noise(data)
            elif noise_type == 'speckle':
                data = add_speckle_noise(data)
            elif noise_type == 'salt_and_pepper':
                data = add_salt_and_pepper_noise(data)
            elif noise_type == 'poisson':
                data = add_poisson_noise(data)
            elif noise_type == 'gaussian_blur':
                data = add_gaussian_blur(data)
            elif noise_type == 'color_jitter':
                data = add_color_jitter(data)

            loss = ddpm.compute_loss(data)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss on {noise_type} noise: {avg_loss}")

def visualize_noise_application(ddpm, noise_type, num_vis_particles=10):
    target_ds = next(iter(dataloader))[0]
    fig, axs = plt.subplots(1, 10, figsize=(28, 3))
    for i, t in enumerate(range(0, 500, 50)):
        x_t = ddpm.q_sample(target_ds[:num_vis_particles].to(ddpm.device), (torch.ones(num_vis_particles) * t).to(ddpm.device))
        x_t = x_t.cpu()
        axs[i].scatter(x_t[:, 0], x_t[:, 1], color="white", edgecolor="gray", s=5)
        axs[i].set_axis_off()
        axs[i].set_title(f"{noise_type} noise, q(x_{t})")
    plt.savefig(f'noise_application_{noise_type}.png')
    plt.show()

def visualize_forward_reverse(ddpm, noise_type, num_vis_particles=10):
    target_ds = next(iter(dataloader))[0]
    fig, axs = plt.subplots(2, 10, figsize=(28, 6))
    for i, t in enumerate(range(0, 500, 50)):
        x_t = ddpm.q_sample(target_ds[:num_vis_particles].to(ddpm.device), (torch.ones(num_vis_particles) * t).to(ddpm.device))
        x_t_reverse = ddpm.p_sample(x_t.to(ddpm.device), (torch.ones(num_vis_particles) * t).to(ddpm.device))

        x_t = x_t.cpu()
        x_t_reverse = x_t_reverse.cpu()

        axs[0, i].scatter(x_t[:, 0], x_t[:, 1], color="white", edgecolor="gray", s=5)
        axs[0, i].set_axis_off()
        axs[0, i].set_title(f"Forward {noise_type} q(x_{t})")

        axs[1, i].scatter(x_t_reverse[:, 0], x_t_reverse[:, 1], color="white", edgecolor="blue", s=5)
        axs[1, i].set_axis_off()
        axs[1, i].set_title(f"Reverse {noise_type} p(x_{t})")
    plt.savefig(f'forward_reverse_{noise_type}.png')
    plt.show()

def main():
    ddpm = DDPM().to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-4)

    noise_types = ['gaussian', 'speckle', 'salt_and_pepper', 'poisson', 'gaussian_blur', 'color_jitter']
    for model_id, noise_type in enumerate(noise_types):
        print(f"Training with {noise_type} noise...")
        train_model(ddpm, dataloader, optimizer, num_epochs=10, noise_type=noise_type, model_id=model_id)
        evaluate_model(ddpm, dataloader, noise_type)
        visualize_noise_application(ddpm, noise_type)
        visualize_forward_reverse(ddpm, noise_type)

if __name__ == "__main__":
    main()
