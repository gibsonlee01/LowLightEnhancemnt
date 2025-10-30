import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import os
import argparse
from data_loaders.lol import lowlight_loader_sh
from torch.utils.tensorboard import SummaryWriter

# -----------------------------
# LightEstimationNet (SH version)
# -----------------------------
class LightEstimationNet(nn.Module):
    def __init__(self, in_channels=3, sh_order=2):
        super(LightEstimationNet, self).__init__()
        num_coeffs = (sh_order + 1) ** 2   # e.g., 9
        out_dim = 3 * num_coeffs           # 27차원

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 512, 1, 1)
            nn.Flatten(),                  # (B, 512)
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)        # (B, 27)
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.fc(feat)   # (B, 27)
        return out

