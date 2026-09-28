import torch
from torch import nn


class WeightedForecastLoss(nn.Module):
    """Emphasize measurable precipitation/echo instead of background pixels."""
    def __init__(self, echo_threshold=0.15, strong_threshold=0.60):
        super().__init__()
        self.echo_threshold = echo_threshold
        self.strong_threshold = strong_threshold

    def forward(self, prediction, target):
        weight = torch.ones_like(target)
        weight = weight + 8.0 * (target >= self.echo_threshold).float()
        weight = weight + 12.0 * (target >= self.strong_threshold).float()
        return ((prediction - target).abs() * weight).mean()
