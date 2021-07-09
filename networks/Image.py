from Params import Params

import torch
import torch.nn as nn

import numpy as np


class StdConvEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.in_ch = in_ch

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=8, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def find_output_size(self, in_spatial_dims):
        test_input = torch.zeros(1, self.in_ch, *in_spatial_dims)#.to(Params.device)
        x = self.net(test_input)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.net(x)
        return torch.flatten(x, start_dim=1)
