# reverse_salt_and_pepper
import torch
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
import datetime
import time
# import requests
from pathlib import Path
from torchvision.utils import make_grid
# from scheduler import linear_schedule
from torchvision.utils import make_grid
from torch.optim import Adam
from noise_salt_and_pepper import reverse_transform_pil, reverse_transform_tensor, sample_q
from unet_model import DiffusionUnet
from reverse_salt_and_pepper import sample_p, sampling
import numpy as np
from torchvision.utils import save_image
# from scheduler import cosine_schedule
from forward import image_size, num_timesteps
from custom_data_loader_celeba import get_celeba_dataloader

# Dataset and DataLoader

# dataset = load_dataset("celeba", split='train')

# def transforms(data):
#   images = [transform(im) for im in data['image']]
#   return {'images': images}

# dataset.set_transform(transforms)
image_size = 64
batch_size=32
# train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_dataloader = get_celeba_dataloader(root='./data/celeba/img_align_celeba', batch_size=batch_size, image_size=image_size)
     

# batch = next(iter(train_dataloader))
# print(#'Shape:', batch['images'].shape,
#       # '\nBounds:', batch['images'].min().item(), 'to', batch['images'].max().item())
# reverse_transform_pil(batch['images'][20])



# Training

# from pathlib import Path
results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
     

def compute_loss(model, x0, t, noise=None):
  if noise is None:
    noise = torch.randn_like(x0)

  x_t = sample_q(x0, t, noise)
  predicted_noise = model(x_t, t)
  loss = F.l1_loss(noise, predicted_noise)
  return loss
     


device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiffusionUnet(dim=image_size, channels=3, dim_mults=(1, 2, 4, 8)).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
     

model_saved_file_path = Path('/checkpoints/saved_model_salt_and_pepper.pth')
if model_saved_file_path.exists():
  saved_model = torch.load(str(model_saved_file_path))
else:
  model_saved_file_path.parent.mkdir(parents=True, exist_ok=True)
  saved_model = {}
  start_epoch = 0
     

if saved_model:
    print('loading model')
    model.load_state_dict(saved_model['model'])
    optimizer.load_state_dict(saved_model['optimizer'])
    current_epoch = saved_model['epoch']
     


epochs = 1000
loss_steps = 50
sample_every = 1000
loss_for_mean = np.zeros(loss_steps)
prev_time = time.time()
for epoch in range(start_epoch, epochs):
  for batch_index, batch in enumerate(train_dataloader):
    # images = batch['images'].to(device)
    images = batch[0].to(device)  # Access images directly from the list
    # sample t according to Algorithm 1
    t = torch.randint(1, num_timesteps, (images.shape[0],), device=device).long()
    loss = compute_loss(model, images, t)
    current_step = batch_index + epoch * len(train_dataloader)
    if current_step % loss_steps == 0:

      # Determine approximate time left
      batches_done = epoch * len(train_dataloader) + batch_index
      batches_left = epochs * len(train_dataloader) - current_step
      time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / loss_steps)
      prev_time = time.time()

      print(f'Loss at epoch {epoch}, batch {batch_index}: {loss_for_mean.mean()} | time remaining: {time_left}')
      loss_for_mean[:] = 0
    loss_for_mean[current_step%loss_steps] = loss.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if current_step % sample_every == 0:
      batch_to_sample = 5
      sample_images_list = sampling(model, (batch_to_sample, 3, image_size, image_size))
      sample_images = torch.cat(sample_images_list, dim=0)
      sample_images = reverse_transform_tensor(sample_images)
      save_image(sample_images, str(results_folder / f'sample_salt_and_pepper_{current_step}.png'), nrow=batch_to_sample)

  saved_model['epoch'] = epoch + 1
  saved_model['model'] = model.state_dict()
  saved_model['optimizer']= optimizer.state_dict()
  torch.save(saved_model, '/checkpoints/saved_model_salt_and_pepper.pth')

# Results

reverse_transform_pil(make_grid(sampling(model, (16, 3, 64, 64))[-1], nrow=8))
image_steps_list = sampling(model, (4, 3, 64, 64), 300)
image_steps = torch.cat(image_steps_list, dim=0)
reverse_transform_pil(make_grid(image_steps, nrow=4))
