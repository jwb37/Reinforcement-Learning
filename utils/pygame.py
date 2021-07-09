import torch
import pygame
import numpy as np

def greyscale_tensor(surface):
    pixels = pygame.surfarray.array3d(surface).astype(np.float32)

    greyscale = pixels.dot([0.298, 0.587, 0.114])
    greyscale = greyscale / 255.0
    greyscale = torch.tensor(greyscale, dtype=torch.float)
    greyscale = greyscale.unsqueeze(0)
    greyscale = greyscale.unsqueeze(0)

    return greyscale

def rgb_tensor(surface):
    pixels = pygame.surfarray.array3d(surface).astype(np.float32)

    pixels = np.moveaxis(pixels, -1, 0)

    return torch.from_numpy(pixels).unsqueeze(0)
