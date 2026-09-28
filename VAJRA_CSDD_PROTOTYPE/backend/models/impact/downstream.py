"""Impact layer kept downstream of meteorological predictions."""
import numpy as np


def compute_impact_risk(hazard_probability, exposure, vulnerability=1.0):
    """Combine meteorological probability with external exposure only."""
    hazard = float(np.clip(hazard_probability, 0.0, 1.0))
    exposed = float(max(exposure, 0.0))
    vuln = float(np.clip(vulnerability, 0.0, 1.0))
    return {"risk_index": hazard * vuln * exposed, "hazard_probability": hazard}
