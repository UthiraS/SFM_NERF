import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import shutil

def read_data(device):
    data = np.load("tiny_nerf_data.npz")
    images = data["images"]
    im_shape = images.shape
    (num_images, H, W, _) = images.shape
    poses = data["poses"]
    poses = torch.from_numpy(poses).to(device)
    focal =  data["focal"]
    focal = torch.from_numpy(focal)
    images = torch.from_numpy(images).to(device)

    return images, poses, focal

def positional_encoding(x, L):
  gamma = [x]
  for i in range(L):
    gamma.append(torch.sin((2.0**i) * x))
    gamma.append(torch.cos((2.0**i) * x))
  gamma = torch.cat(gamma, axis = -1)
  return gamma


def mini_batches(inputs, batch_size):
  return [inputs[i:i + batch_size] for i in range(0, inputs.shape[0], batch_size)]

def plot_figures(Epochs, log_loss):
    plt.figure(figsize=(10, 4))
    plt.plot(Epochs, log_loss)
    plt.title("loss")
    # plt.show()
    plt.savefig("Results/Loss.pdf", format="pdf")

def make_video(fps, path, video_file):
  print("Creating video {}, FPS={}".format(video_file, fps))
  clip = ImageSequenceClip(path, fps = fps)
  clip.write_videofile(video_file)
  shutil.rmtree(path)

def get_rays(h, w, f, pose, near_threshold, far_threshold, Nc, device):
  x = torch.linspace(0, w-1, w)
  y = torch.linspace(0, h-1, h)
  xi, yi = torch.meshgrid(x, y, indexing='xy')
  xi = xi.to(device)
  yi = yi.to(device)
  norm_x = (xi - w * 0.5) / f
  norm_y = (yi - h * 0.5) / f
  directions = torch.stack([norm_x, - norm_y, -torch.ones_like(xi)], dim = -1)
  directions = directions[..., None,:]
  rotation = pose[:3, :3]
  translation = pose[:3, -1]

  camera_directions = directions * rotation
  ray_directions = torch.sum(camera_directions, dim = -1)
  ray_directions = ray_directions/torch.linalg.norm(ray_directions, dim = -1, keepdims = True)
  ray_origins =  torch.broadcast_to(translation, ray_directions.shape)

  depth_val = torch.linspace(near_threshold, far_threshold, Nc)
  noise_shape = list(ray_origins.shape[:-1]) + [Nc]
  noise = torch.rand(size = noise_shape) * (far_threshold - near_threshold)/Nc
  depth_val = depth_val + noise
  depth_val = depth_val.to(device)
  query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_val[..., :, None]

  return ray_directions, ray_origins, depth_val, query_points

def render(radiance_field, ray_origins, depth_values):
  sigma_a = F.relu(radiance_field[...,3])  
  rgb = torch.sigmoid(radiance_field[...,:3]) 
  one_e_10 = torch.tensor([1e10], dtype = ray_origins.dtype, device = ray_origins.device)
  dists = torch.cat((depth_values[...,1:] - depth_values[...,:-1], one_e_10.expand(depth_values[...,:1].shape)), dim = -1)
  alpha = 1. - torch.exp(-sigma_a * dists)       
  weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)  
  rgb_map = (weights[..., None] * rgb).sum(dim = -2)        
  depth_map = (weights * depth_values).sum(dim = -1)
  acc_map = weights.sum(-1)
  return rgb_map, depth_map, acc_map

def cumprod_exclusive(tensor) :
  dim = -1
  cumprod = torch.cumprod(tensor, dim)
  cumprod = torch.roll(cumprod, 1, dim)
  cumprod[..., 0] = 1.
  
  return cumprod