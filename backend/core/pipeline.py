import numpy as np
from models.consistency.object_field import score
from models.data_quality.quality import overall_quality
from models.field_dynamics.eulerian import advect
from models.hazard_heads.probability import probabilities
from models.object_dynamics.lagrangian import forecast as object_forecast
from models.state_estimation import estimate
from models.storm_detection import detect
from models.storm_lifecycle.events import LifecycleTracker
from models.uncertainty import ensemble
from models.storm_tracking import track
from models.system import VAJRACSDDModel


def run(current_field, previous_field, sensor_quality, lead_minutes=60):
    current = detect(current_field)
    previous = detect(previous_field) if previous_field is not None else []
    tracked = track(current, previous)
    state = estimate(tracked, sensor_quality)
    objects = object_forecast(state, lead_minutes)
    field = advect(
        current_field,
        np.mean([o.get("dx", 0) for o in state] or [0]),
        np.mean([o.get("dy", 0) for o in state] or [0]),
        lead_minutes,
    )
    consistency = score(objects, field)
    tracker = LifecycleTracker()
    tracker.previous = previous
    _, events = tracker.update(current)
    ens = ensemble(field, members=24, noise=0.035)
    max_intensity = max([o["intensity"] for o in objects] or [0])
    return {
        "objects": objects, "field": field, "events": events,
        "consistency": consistency, "quality": overall_quality(sensor_quality),
        "hazards": probabilities(max_intensity),
        "probabilistic": {
            "members": ens["members"], "p10": ens["p10"],
            "p50": ens["p50"], "p90": ens["p90"], "spread": ens["spread"]
        },
    }


def architecture_components():
    return VAJRACSDDModel()
