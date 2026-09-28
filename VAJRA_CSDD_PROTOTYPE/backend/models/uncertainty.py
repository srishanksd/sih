"""Lightweight ensemble uncertainty baseline."""
import numpy as np


def ensemble(field, members=12, noise=0.035, seed=84):
    rng = np.random.default_rng(seed)
    samples = np.stack([np.clip(field + rng.normal(0, noise, field.shape), 0, 1) for _ in range(members)])
    return {"mean": samples.mean(0), "p10": np.quantile(samples, .1, 0), "p90": np.quantile(samples, .9, 0), "spread": samples.std(0)}
