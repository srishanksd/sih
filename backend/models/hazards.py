"""Hazard probability mapping from normalized convective intensity."""
def hazards(intensity):
    x = max(0.0, min(1.0, float(intensity)))
    return {"lightning": min(1, x*1.05), "hail": max(0, x-.25)/.75, "wind": max(0, x-.35)/.65, "cloudburst": max(0, x-.55)/.45}
