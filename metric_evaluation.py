import torch
import matplotlib.pyplot as plt
import numpy as np
from Diffusion_combined import DDPM, add_gaussian_noise, add_speckle_noise, add_salt_and_pepper_noise, add_poisson_noise, add_gaussian_blur, add_color_jitter
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr, mean_absolute_error as mae
from lpips import LPIPS
import scoot

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# Load sample image
sample_image_path = 'sample_image.png'
sample_image = Image.open(sample_image_path).convert('RGB')
sample_image = transform(sample_image).unsqueeze(0)
sample_image_np = sample_image.squeeze(0).permute(1, 2, 0).numpy()

# Load the best models
noise_types = ['gaussian', 'speckle', 'salt_and_pepper', 'poisson', 'gaussian_blur', 'color_jitter']
best_models = {}
ddpm = DDPM().to('cuda' if torch.cuda.is_available() else 'cpu')

for noise_type in noise_types:
    checkpoint_path = f'checkpoints/saved_model_{noise_type}.pth'
    if os.path.exists(checkpoint_path):
        ddpm.load_state_dict(torch.load(checkpoint_path))
        best_models[noise_type] = ddpm

# Evaluate images
lpips_model = LPIPS(net='alex').cuda()
evaluation_results = {}
generated_images = []

for noise_type, model in best_models.items():
    with torch.no_grad():
        sample_output = model(sample_image.cuda()).cpu().clamp(0, 1)
        generated_images.append(sample_output)
        save_image(sample_output, f'{noise_type}_output.png')

        sample_output_np = sample_output.squeeze(0).permute(1, 2, 0).numpy()

        # Metrics calculation
        ssim_value = ssim(sample_image_np, sample_output_np, multichannel=True)
        psnr_value = psnr(sample_image_np, sample_output_np)
        mae_value = mae(sample_image_np, sample_output_np)
        lpips_value = lpips_model(sample_image.cuda(), sample_output.cuda()).item()
        scoot_value = scoot.calculate_scoot(sample_image_np, sample_output_np)

        evaluation_results[noise_type] = {
            'SSIM': ssim_value,
            'PSNR': psnr_value,
            'MAE': mae_value,
            'LPIPS': lpips_value,
            'SCOOT': scoot_value
        }

# Visualization
fig, axs = plt.subplots(1, len(noise_types) + 1, figsize=(20, 5))
axs[0].imshow(sample_image_np)
axs[0].set_title('Original')
axs[0].axis('off')

for i, (noise_type, img) in enumerate(zip(noise_types, generated_images)):
    img_np = img.squeeze(0).permute(1, 2, 0).numpy()
    axs[i + 1].imshow(img_np)
    axs[i + 1].set_title(noise_type.capitalize())
    axs[i + 1].axis('off')

plt.savefig('image_comparison.png')
plt.show()

# Display results
import pandas as pd

evaluation_df = pd.DataFrame(evaluation_results).T
print(evaluation_df)
evaluation_df.to_csv('evaluation_results.csv')

# Identify best metrics
best_metrics = {}

for metric in evaluation_df.columns:
    if metric in ['LPIPS']:  # Smaller is better # 'FID', 'C-FID', 
        best_noise_type = evaluation_df[metric].idxmin()
    else:  # Bigger is better
        best_noise_type = evaluation_df[metric].idxmax()
    best_metrics[metric] = best_noise_type

print("Best Models per Metric:")
print(best_metrics)
