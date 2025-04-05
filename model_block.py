# Model Bulding Blocks
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightStandardizedConv2d(nn.Conv2d):
  """
  https://arxiv.org/abs/1903.10520
  weight standardization purportedly works synergistically with group normalization
  """

  def forward(self, x):
    eps = 1e-5 if x.dtype == torch.float32 else 1e-3

    weight = self.weight
    mean = weight.mean(dim=[1,2,3], keepdim=True)
    variance = weight.var(dim=[1,2,3], keepdim=True, correction=0)
    normalized_weight = (weight - mean) / torch.sqrt(variance)

    return F.conv2d(
        x,
        normalized_weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups
    )


class Block(nn.Module):
  def __init__(self, in_channels, out_channels, groups=8):
    super().__init__()
    self.proj = WeightStandardizedConv2d(in_channels, out_channels, 3, padding=1)
    self.norm = nn.GroupNorm(groups, out_channels)
    self.act = nn.SiLU()

  def forward(self, x, scale_shift=None):
    x = self.proj(x)
    x = self.norm(x)

    if scale_shift is not None:
      scale, shift = scale_shift
      x = x * (scale + 1) + shift

    x = self.act(x)
    return x

class ResnetBlock(nn.Module):
  def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
    super().__init__()
    if time_emb_dim is not None:
      self.mlp = nn.Sequential(
          nn.SiLU(),
          nn.Linear(time_emb_dim, 2 * out_channels)
      )
    else:
      self.mlp = None

    self.block1 = Block(in_channels, out_channels, groups)
    self.block2 = Block(out_channels, out_channels, groups)
    if in_channels == out_channels:
      self.res_conv = nn.Identity()
    else:
      self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

  def forward(self, x, time_emb=None):
    scale_shift = None
    if self.mlp is not None and time_emb is not None:
      time_emb = self.mlp(time_emb)
      time_emb = time_emb.view(*time_emb.shape, 1, 1)
      scale_shift = time_emb.chunk(2, dim=1) ########

    h = self.block1(x, scale_shift=scale_shift)
    h = self.block2(h)
    return h + self.res_conv(x)

     

class Attention(nn.Module):
  def __init__(self, in_channels, num_heads=4, dim_head=32):
    super().__init__()
    self.num_heads = num_heads
    self.dim_head = dim_head
    self.scale_factor = 1 / (dim_head) ** 0.5
    self.hidden_dim = num_heads * dim_head
    self.input_to_qkv = nn.Conv2d(in_channels, 3 * self.hidden_dim, 1, bias=False)
    self.to_output = nn.Conv2d(self.hidden_dim, in_channels, 1)

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.input_to_qkv(x)
    q, k, v = map(lambda t: t.view(b, self.num_heads, self.dim_head, h * w), qkv.chunk(3, dim=1))
    q = q * self.scale_factor
    # dot product between the columns of q and k
    sim = torch.einsum("b h c i, b h c j -> b h i j", q, k)
    sim = sim - sim.amax(dim=-1, keepdim=True).detach()
    attention = sim.softmax(dim=-1)

    # dot product between the rows to get the wighted values as columns
    output = torch.einsum("b h i j, b h c j -> b h i c", attention, v)
    output = output.permute(0, 1, 3, 2).reshape((b, self.hidden_dim, h, w))
    return self.to_output(output)


class LinearAttention(nn.Module):
  def __init__(self, in_channels, num_heads=4, dim_head=32):
    super().__init__()
    self.num_heads = num_heads
    self.dim_head = dim_head
    self.scale_factor = 1 / (dim_head) ** 0.5
    self.hidden_dim = num_heads * dim_head
    self.input_to_qkv = nn.Conv2d(in_channels, 3 * self.hidden_dim, 1, bias=False)
    self.to_output = nn.Sequential(
        nn.Conv2d(self.hidden_dim, in_channels, 1),
        nn.GroupNorm(1, in_channels)
    )

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.input_to_qkv(x)
    q, k, v = map(lambda t: t.view(b, self.num_heads, self.dim_head, h * w), qkv.chunk(3, dim=1))

    q = q.softmax(dim=-2)
    k = k.softmax(dim=-1)

    q = q * self.scale_factor
    context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
    output = torch.einsum("b h d e, b h d n -> b h e n", context, q)
    output = output.view((b, self.hidden_dim, h, w))
    return self.to_output(output)
     

class PreGroupNorm(nn.Module):
  def __init__(self, dim , func, groups=1):
    super().__init__()
    self.func = func
    self.group_norm = nn.GroupNorm(groups, dim)

  def forward(self, x):
    x = self.group_norm(x)
    x = self.func(x)
    return x