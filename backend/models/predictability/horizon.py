"""Predictability-horizon estimation from ensemble spread."""
import torch


def estimate_predictability(spread, threshold=0.20, step_minutes=15):
    """Return first lead index where uncertainty exceeds the threshold."""
    # spread: [B,T,...] or [T,...].
    if spread.ndim < 2:
        raise ValueError("spread must contain a time dimension")
    time_dim = spread.ndim - 1 if spread.ndim == 2 else 1
    score = spread.reshape(spread.shape[0], spread.shape[1], -1).mean(-1) if spread.ndim > 2 else spread
    horizon = []
    for row in score.detach().cpu():
        hit = torch.where(row >= threshold)[0]
        idx = int(hit[0]) if len(hit) else len(row)
        horizon.append(idx * step_minutes)
    return horizon
