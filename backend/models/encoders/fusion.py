"""Reliability-aware cross-modal fusion for radar/satellite/lightning/NWP inputs."""
import torch
from torch import nn


class ReliabilityAwareFusion(nn.Module):
    """Fuse modality embeddings while suppressing unreliable sensors."""
    def __init__(self, d_model=128, heads=4, modalities=5):
        super().__init__()
        self.modalities = modalities
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(modalities, d_model), nn.Sigmoid())

    def forward(self, embeddings, quality):
        # embeddings: [B,M,D], quality: [B,M] in [0,1].
        if embeddings.ndim != 3 or quality.ndim != 2:
            raise ValueError("Expected embeddings [B,M,D] and quality [B,M]")
        weights = self.gate(quality).unsqueeze(1)
        tokens = embeddings * weights.transpose(1, 2)
        q = self.query.expand(embeddings.size(0), -1, -1)
        fused, _ = self.attn(q, tokens, tokens)
        return self.norm(fused[:, 0] + embeddings.mean(dim=1))
