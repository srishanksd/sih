"""Multi-hazard probability heads conditioned on the forecast state."""
import torch
from torch import nn


class MultiHazardHeads(nn.Module):
    """Predict hazard probabilities independently from a shared state."""
    HAZARDS = ("thunderstorm", "lightning", "hail", "cloudburst", "wind")

    def __init__(self, state_dim=128, hazards=None):
        super().__init__()
        self.hazards = tuple(hazards or self.HAZARDS)
        self.heads = nn.ModuleDict({name: nn.Linear(state_dim, 1) for name in self.hazards})

    def forward(self, state):
        return {name: torch.sigmoid(head(state)).squeeze(-1) for name, head in self.heads.items()}
