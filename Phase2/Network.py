import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def training(height, width, focal, pose, near_threshold, far_threshold, Nc, batch_size, N_encode, model, device):
    
    ray_directions, ray_origins, depth_values, query_points = get_rays(height, width, focal, pose, near_threshold, far_threshold, Nc, device)
    flat_query_points = query_points.reshape((-1,3))
    encoded_points = positional_encoding(flat_query_points, N_encode)
    batches = mini_batches(encoded_points, batch_size = batch_size)
    predictions = []
    for batch in batches:
        predictions.append((model(batch)))

    radiance_field_flat = torch.cat(predictions, dim=0)
    unflat_shape = list(query_points.shape[:-1]) + [4]
    radiance_field = torch.reshape(radiance_field_flat, unflat_shape)
    logits_rgb, _, _ = render(radiance_field, ray_directions, depth_values)

    return logits_rgb

class Nerf(nn.Module):

  def __init__(self, filter_size = 128, N_encode = 6):
    super(Nerf, self).__init__()
    self.layer1 = nn.Linear(3+3*2*N_encode, filter_size)
    self.layer2 = nn.Linear(filter_size, filter_size)
    self.layer3 = nn.Linear(filter_size, 4)
  
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = self.layer3(x)
    return x