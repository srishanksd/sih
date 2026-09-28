"""Encode fused observations into an explicit evolving convective state."""
import torch
from torch import nn


class ConvectiveStateEncoder(nn.Module):
    """Temporal state estimator producing interpretable state vectors."""
    def __init__(self, input_dim=128, state_dim=128):
        super().__init__()
        self.proj = nn.Linear(input_dim, state_dim)
        self.gru = nn.GRU(state_dim, state_dim, batch_first=True)
        self.state_head = nn.Sequential(nn.LayerNorm(state_dim), nn.Tanh())

    def forward(self, fused_sequence, hidden=None):
        # fused_sequence: [B,T,D].
        x = self.proj(fused_sequence)
        sequence, hidden = self.gru(x, hidden)
        state = self.state_head(sequence[:, -1])
        return state, sequence, hidden
