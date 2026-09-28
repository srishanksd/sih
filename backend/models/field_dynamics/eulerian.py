from scipy.ndimage import shift


def advect(field, dx, dy, lead_minutes):
    steps = float(lead_minutes) / 15.0
    return shift(field, (dy*steps, dx*steps), mode="nearest")
