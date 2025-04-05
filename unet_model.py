import torch
import torch.nn as nn
     
from model_block import ResnetBlock, PreGroupNorm, LinearAttention, Attention
from helper import SpaceToDepth, Residual, upsample, downsample
     
# Position embedding

class SinusodialPositionEmbedding(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim

  def forward(self, time_steps):
    positions = torch.unsqueeze(time_steps, 1)
    half_dim = self.embedding_dim // 2
    embeddings = torch.zeros((time_steps.shape[0], self.embedding_dim), device=time_steps.device)
    denominators = 10_000 ** (2 * torch.arange(self.embedding_dim // 2, device=time_steps.device) / self.embedding_dim)
    embeddings[:, 0::2] = torch.sin(positions/denominators)
    embeddings[:, 1::2] = torch.cos(positions/denominators)
    return embeddings


# The Unet Model

class DiffusionUnet(nn.Module):
  def __init__(self, dim, init_dim=None, output_dim=None, dim_mults=(1, 2, 4, 8), channels=3, resnet_block_groups=4):
    super().__init__()

    self.channels = channels
    init_dim = init_dim if init_dim is not None else dim
    self.init_conv = nn.Conv2d(self.channels, init_dim, 1)
    dims = [init_dim] + [m * dim for m in dim_mults]
    input_output_dims = list(zip(dims[:-1], dims[1:]))

    time_dim = 4 * dim  # time embedding

    self.time_mlp = nn.Sequential(
        SinusodialPositionEmbedding(dim),
        nn.Linear(dim, time_dim),
        nn.GELU(),
        nn.Linear(time_dim, time_dim)
    )

    # down layers
    self.down_layers = nn.ModuleList([])
    for ii, (dim_in, dim_out) in enumerate(input_output_dims, 1):
      is_last = ii == len(input_output_dims)
      self.down_layers.append(
          nn.ModuleList(
              [
                  ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                  ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                  Residual(PreGroupNorm(dim_in, LinearAttention(dim_in))),
                  downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
              ]
          )
      )

      # middle layers
      mid_dim = dims[-1]
      self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
      self.mid_attention = Residual(PreGroupNorm(mid_dim, Attention(mid_dim)))
      self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

      # up layers
      self.up_layers = nn.ModuleList([])
      for ii, (dim_in, dim_out) in enumerate(reversed(input_output_dims), 1):
        is_last = ii == len(input_output_dims)
        self.up_layers.append(
            nn.ModuleList(
                [
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                    Residual(PreGroupNorm(dim_out, LinearAttention(dim_out))),
                    upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ]
            )
        )

        self.output_dim = output_dim if output_dim is not None else channels
        self.final_res_block = ResnetBlock(2 * dim, dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, self.output_dim, 1)

  def forward(self, x, time):
    x = self.init_conv(x)
    init_result = x.clone()
    t = self.time_mlp(time)
    h = []

    for block1, block2, attention, downsample_block in self.down_layers:
      x = block1(x, t)
      h.append(x)

      x = block2(x, t)
      x = attention(x)

      h.append(x)

      x = downsample_block(x)

    x = self.mid_block1(x, t)
    x = self.mid_attention(x)
    x = self.mid_block2(x ,t)

    for block1, block2, attention, upsample_block in self.up_layers:
      x = torch.cat((x , h.pop()), dim=1)
      x = block1(x, t)

      x = torch.cat((x, h.pop()), dim=1)
      x = block2(x, t)

      x = attention(x)

      x = upsample_block(x)

    x = torch.cat((x, init_result), dim=1)
    x = self.final_res_block(x, t)
    x = self.final_conv(x)
    return x