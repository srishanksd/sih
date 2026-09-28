"""Nearest-neighbour storm-object tracker baseline."""
import math


def track(current, previous=None):
    previous = previous or []
    tracks = []
    for obj in current:
        if previous:
            p = min(previous, key=lambda q: math.hypot(q["x"]-obj["x"], q["y"]-obj["y"]))
            dx, dy = obj["x"]-p["x"], obj["y"]-p["y"]
            speed = math.hypot(dx, dy)
        else:
            dx = dy = speed = 0.0
        tracks.append({**obj, "dx": dx, "dy": dy, "speed": speed})
    return tracks
