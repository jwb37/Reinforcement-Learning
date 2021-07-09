import torch
import numpy as np
import torch.nn as nn

from collections import namedtuple

InceptionParams = namedtuple('InceptionParams', ['in_dim', 'conv1', 'reduce3', 'conv3', 'reduce5', 'conv5', 'pool'])


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.net(x)


class InceptionBlockv1(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.conv1 = ConvBlock(params.in_dim, params.conv1, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Sequential(
            ConvBlock(params.in_dim, params.reduce3, kernel_size=1, stride=1, padding=0),
            ConvBlock(params.reduce3, params.conv3, kernel_size=3, stride=1, padding=1),
        )
        self.conv5 = nn.Sequential(
            ConvBlock(params.in_dim, params.reduce5, kernel_size=1, stride=1, padding=0),
            ConvBlock(params.reduce5, params.conv5, kernel_size=5, stride=1, padding=2)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(params.in_dim, params.pool, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return torch.cat(
            (
                self.conv1(x),
                self.conv3(x),
                self.conv5(x),
                self.pool(x)
            ),
            dim = 1
        )


class InceptionConvEncoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.in_ch = in_ch

        self.net = nn.Sequential(
            nn.Conv2d( in_ch, 64, kernel_size=7, stride=2, padding=3 ),
            nn.MaxPool2d(2, stride=2),
            InceptionBlockv1( InceptionParams(64, 32, 48, 64, 8, 16, 16) ),
            InceptionBlockv1( InceptionParams(128, 32, 48, 64, 8, 16, 16) ),
            nn.MaxPool2d(2, stride=2),
            InceptionBlockv1( InceptionParams(128, 64, 96, 128, 16, 32, 32) ),
            InceptionBlockv1( InceptionParams(256, 64, 96, 128, 16, 32, 32) ),
            nn.MaxPool2d(2, stride=2),
            InceptionBlockv1( InceptionParams(256, 192, 96, 208, 16, 48, 64) ),
            InceptionBlockv1( InceptionParams(512, 160, 112, 224, 24, 64, 64) ),
            nn.AvgPool2d(2, stride=2)
        )

    def find_output_size(self, in_spatial_dims):
        test_input = torch.zeros(1, self.in_ch, *in_spatial_dims)
        x = self.net(test_input)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = self.net(x)
        return torch.flatten(x, start_dim=1)
