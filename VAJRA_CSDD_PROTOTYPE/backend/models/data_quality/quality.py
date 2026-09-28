import numpy as np

REQUIRED = ("DWR", "INSAT", "ILDN", "AWS", "GFS")

def normalize_quality(values):
    return {name: float(np.clip(values.get(name, 0.0), 0.0, 1.0)) for name in REQUIRED}

def modality_mask(values, threshold=0.25):
    q = normalize_quality(values)
    return {name: int(score >= threshold) for name, score in q.items()}

def overall_quality(values):
    q = normalize_quality(values)
    active = [v for v in q.values() if v > 0]
    return float(np.mean(active)) if active else 0.0
