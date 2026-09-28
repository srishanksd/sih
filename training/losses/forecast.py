import torch
from torch import nn


class WeightedForecastLoss(nn.Module):
    """Pixel loss with extra weight for strong convective echoes."""
    def __init__(self, strong_threshold=0.6, strong_weight=2.0):
        super().__init__()
        self.strong_threshold = strong_threshold
        self.strong_weight = strong_weight

    def forward(self, prediction, target):
        weight = 1.0 + self.strong_weight * (target >= self.strong_threshold).float()
        return ((prediction - target).abs() * weight).mean()
