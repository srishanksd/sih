"""Quality-aware multimodal state representation baseline."""
def estimate(storms, sensor_quality):
    q = sum(sensor_quality.values()) / max(len(sensor_quality), 1)
    for s in storms:
        s["state"] = {
            "intensity": s["intensity"],
            "motion": [s["dx"], s["dy"]],
            "quality": q,
            "growth_proxy": s["intensity"] * q,
        }
    return storms
