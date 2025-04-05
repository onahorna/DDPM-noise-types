import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime
import time

# Helper function

def space_to_depth(x, size=2):
    """
    Downsacle method that use the depth dimension to
    downscale the spatial dimensions
    Args:
        x (torch.Tensor): a tensor to downscale
        size (float): the scaling factor

    Returns:
        (torch.Tensor): new spatial downscale tensor
    """
    b, c, h, w = x.shape
    out_h = h // size
    out_w = w // size
    out_c = c * (size * size)
    x = x.reshape((-1, c, out_h, size, out_w, size))
    x = x.permute((0, 1, 3, 5, 2, 4))
    x = x.reshape((-1, out_c, out_h, out_w))
    return x


class SpaceToDepth(nn.Module):
  def __init__(self, size):
    super().__init__()
    self.size = size

  def forward(self, x):
    return space_to_depth(x, self.size)


class Residual(nn.Module):
  """
  Apply residual connection using an input function
  Args:
    func (function): a function to apply over the input
  """
  def __init__(self, func):
    super().__init__()
    self.func = func

  def forward(self, x, *args, **kwargs):
    return x + self.func(x, *args, **kwargs)

def upsample(in_channels, out_channels=None):
  out_channels = in_channels if out_channels is None else out_channels
  seq = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.Conv2d(in_channels, out_channels, 3, padding=1)
  )
  return seq

def downsample(in_channels, out_channels=None):
  out_channels = in_channels if out_channels is None else out_channels
  seq = nn.Sequential(
      SpaceToDepth(2),
      nn.Conv2d(4 * in_channels, out_channels, 1)
  )
  return seq