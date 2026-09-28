import numpy as np


def probabilities(intensity, uncertainty=0.05):
    x = float(np.clip(intensity, 0, 1))
    return {"thunderstorm": x, "lightning": np.clip(1.15*x,0,1), "hail": np.clip((x-.35)/.65,0,1), "cloudburst": np.clip((x-.6)/.4,0,1), "uncertainty": float(uncertainty)}
