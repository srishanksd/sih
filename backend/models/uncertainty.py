"""Probabilistic ensemble utilities for prototype nowcasting."""
import numpy as np


def ensemble(field, members=24, noise=0.035, seed=84):
    """Fast field ensemble; use trained-model members for operational calibration later."""
    rng = np.random.default_rng(seed)
    samples = np.stack([np.clip(field + rng.normal(0, noise, field.shape), 0, 1)
                        for _ in range(members)])
    return {"members": samples, "mean": samples.mean(0),
            "p10": np.quantile(samples, .10, axis=0),
            "p50": np.quantile(samples, .50, axis=0),
            "p90": np.quantile(samples, .90, axis=0),
            "spread": samples.std(0)}


def summarize_members(samples):
    samples = np.asarray(samples)
    return {"mean": samples.mean(0), "p10": np.quantile(samples,.10,axis=0),
            "p50": np.quantile(samples,.50,axis=0), "p90": np.quantile(samples,.90,axis=0),
            "spread": samples.std(0), "members": int(samples.shape[0])}


def exceedance_probability(samples, threshold):
    return np.mean(np.asarray(samples) >= threshold, axis=0)
