"""Baseline storm-cell detector; replaceable with learned segmentation later."""
import numpy as np
from scipy import ndimage


def detect(field: np.ndarray, threshold: float = 0.62):
    mask = field >= threshold
    labels, count = ndimage.label(mask)
    objects = []
    for i in range(1, count + 1):
        ys, xs = np.where(labels == i)
        if len(xs) < 6:
            continue
        objects.append({"id": f"S{i}", "x": float(xs.mean()), "y": float(ys.mean()), "area": len(xs), "intensity": float(field[ys, xs].mean())})
    return objects
