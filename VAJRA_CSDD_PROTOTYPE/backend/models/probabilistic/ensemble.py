"""Stochastic rollout wrapper for forecast uncertainty."""
import torch
from torch import nn


class StochasticEnsemble(nn.Module):
    """Sample forecast members by perturbing the latent state."""
    def __init__(self, state_dim=128, noise_scale=0.05):
        super().__init__()
        self.noise_scale = noise_scale
        self.scale = nn.Parameter(torch.ones(state_dim))

    def sample_states(self, state, members=8):
        if members < 1:
            raise ValueError("members must be >= 1")
        noise = torch.randn(members, *state.shape, device=state.device)
        return state.unsqueeze(0) + noise * self.noise_scale * self.scale.view(1, 1, -1)

    @staticmethod
    def summarize(samples):
        return {"mean": samples.mean(0), "std": samples.std(0),
                "p10": torch.quantile(samples, 0.10, dim=0),
                "p90": torch.quantile(samples, 0.90, dim=0)}
