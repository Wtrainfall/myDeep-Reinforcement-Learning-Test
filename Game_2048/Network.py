import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn   = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, features_dim), nn.ReLU()
        )
    def forward(self, observations):
        x = observations.view(-1, 1, 4, 4).float()
        return self.net(x)