import numpy as np


def forecast(objects, lead_minutes, growth_gain=0.08):
    lead = float(lead_minutes) / 15.0
    result = []
    for o in objects:
        result.append({**o, "x": o["x"] + o.get("dx", 0)*lead, "y": o["y"] + o.get("dy", 0)*lead,
                       "intensity": float(np.clip(o["intensity"] + growth_gain*o.get("growth_proxy", 0)*lead, 0, 1))})
    return result
