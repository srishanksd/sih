"""Dual-dynamics baseline: object advection + field advection."""
from scipy.ndimage import shift


def lagrangian(storms, steps=24):
    return [[{**s, "x": s["x"] + s["dx"]*k, "y": s["y"] + s["dy"]*k} for s in storms] for k in range(1, steps+1)]

def eulerian(field, dx=0.0, dy=0.0, steps=24):
    return [shift(field, (dy*k, dx*k), mode="nearest") for k in range(1, steps+1)]
