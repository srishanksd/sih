"""Generate a self-contained prototype replay from the trained radar model."""
from pathlib import Path
import sys
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.models.convlstm import ConvLSTMNowcaster
from backend.models.storm_detection import detect
from backend.models.storm_tracking import track
from backend.models.hazard_heads.probability import probabilities

ROOT = Path(__file__).resolve().parents[1]
EVENT_DIR = ROOT / "data" / "processed" / "events"
OUT = ROOT / "data" / "demo"
CHECKPOINT = ROOT / "training" / "checkpoints" / "best.pt"


def main():
    events = sorted(EVENT_DIR.glob("segment_*.npy"))
    if not events:
        raise RuntimeError("No processed event sequences found.")
    # Latest held-out event is used as the prototype replay.
    event = events[-1]
    arr = np.load(event)
    if arr.shape[0] < 12:
        raise RuntimeError("Demo event needs at least 12 frames.")

    x = torch.from_numpy(arr[:8]).float()[None, :, None].cuda()
    actual = arr[8:12]

    checkpoint = torch.load(CHECKPOINT, map_location="cuda")
    model = ConvLSTMNowcaster(1, (32, 64), 1, 3).cuda()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        forecast = model(x, 4)[0, :, 0].cpu().numpy()

    # Use the observed final input frame for the t=0 dashboard state.
    observed = arr[7]
    prev = arr[6]
    current_objects = track(detect(observed), detect(prev))
    for obj in current_objects:
        obj["state"] = {
            "quality": 0.88,
            "growth_proxy": obj["intensity"] * 0.88,
        }

    metadata = {
        "mode": "prototype_replay",
        "source_event": event.name,
        "lead_minutes": [0, 15, 30, 45, 60],
        "input_frames": 8,
        "forecast_frames": 4,
        "model": "ConvLSTM radar baseline",
        "note": "Prototype replay using held-out DWR event; not live data.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "observed.npy", observed.astype(np.float32))
    np.save(OUT / "forecast.npy", forecast.astype(np.float32))
    np.save(OUT / "actual_future.npy", actual.astype(np.float32))

    forecasts = []
    for i, frame in enumerate(forecast):
        storms = track(detect(frame), current_objects)
        intensity = max([s["intensity"] for s in storms] or [float(frame.max())])
        forecasts.append({
            "lead_minutes": (i + 1) * 15,
            "storms": storms,
            "hazards": probabilities(intensity),
            "max_reflectivity_normalized": float(frame.max()),
        })

    metadata["observed_storms"] = current_objects
    metadata["forecast"] = forecasts
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Demo generated from {event.name}")
    print(f"Observed storms: {len(current_objects)}")
    print("Forecast leads: 15, 30, 45, 60 min")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
