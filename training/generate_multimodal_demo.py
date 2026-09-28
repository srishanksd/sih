"""Generate dashboard replay from the trained DWR + INSAT model."""
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

EVENT_DIR = ROOT / "data" / "processed" / "multimodal"
OUT = ROOT / "data" / "demo"
CHECKPOINT = ROOT / "training" / "checkpoints_multimodal" / "best.pt"

def main():
    events = sorted(EVENT_DIR.glob("segment_*.npy"))
    if len(events) < 3:
        raise RuntimeError("Need at least 3 multimodal event sequences.")
    event = events[-1]
    arr = np.load(event).astype(np.float32)
    if arr.shape[0] < 12:
        raise RuntimeError("Demo event needs at least 12 frames.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(arr[:8])[None].to(device)
    actual = arr[8:12, 0]
    observed = arr[7, 0]
    insat_current = arr[7, 1:4]

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = ConvLSTMNowcaster(4, (32, 64), 1, 3).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        forecast = model(x, 4)[0, :, 0].cpu().numpy()

    prev = arr[6, 0]
    current_objects = track(detect(observed), detect(prev))
    for obj in current_objects:
        obj["state"] = {
            "quality": 0.88,
            "growth_proxy": obj["intensity"] * 0.88,
        }

    mae = float(np.mean(np.abs(forecast - actual)))
    persistence = float(np.mean(np.abs(observed[None] - actual)))

    metadata = {
        "mode": "prototype_replay",
        "source_event": event.name,
        "lead_minutes": [0, 15, 30, 45, 60],
        "input_frames": 8,
        "forecast_frames": 4,
        "model": "DWR + INSAT-3DR CTP/CTT ConvLSTM",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "heldout_mae": mae,
        "persistence_mae": persistence,
        "note": "Historical Cherrapunji DWR replay. INSAT is used as an auxiliary input; forecasts are autoregressive radar forecasts.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "observed.npy", observed.astype(np.float32))
    np.save(OUT / "forecast.npy", forecast.astype(np.float32))
    np.save(OUT / "actual_future.npy", actual.astype(np.float32))
    np.save(OUT / "insat_current.npy", insat_current.astype(np.float32))

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
    metadata["insat"] = {
        "channels": ["CTP_cloud_top_signal", "CTT_cold_top_signal", "valid_mask"],
        "valid_fraction": float(insat_current[2].mean()),
    }
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Demo generated from {event.name}")
    print(f"checkpoint_epoch={checkpoint['epoch']}")
    print(f"heldout_MAE={mae:.6f}")
    print(f"persistence_MAE={persistence:.6f}")
    print(f"forecast_max={float(forecast.max()):.4f}")
    print(f"INSAT valid fraction={float(insat_current[2].mean()):.3f}")

if __name__ == "__main__":
    main()
