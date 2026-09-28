"""Promote the evaluated GFS replay into the dashboard demo directory."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "demo_gfs"
DST = ROOT / "data" / "demo"
DST.mkdir(exist_ok=True)

metrics = json.loads((SRC / "metrics.json").read_text())
forecast = np.load(SRC / "forecast.npy")
actual = np.load(SRC / "actual_future.npy")
observed = np.load(SRC / "observed.npy")

if observed.ndim != 2 or forecast.ndim != 3 or actual.ndim != 3:
    raise ValueError(
        f"Unexpected replay shapes: observed={observed.shape}, "
        f"forecast={forecast.shape}, actual={actual.shape}"
    )

meta = {
    "source_event": "chronological held-out GFS event",
    "model": "DWR + INSAT-3DR CTP/CTT + GFS ConvLSTM",
    "checkpoint_epoch": metrics["checkpoint_epoch"],
    "heldout_mae": metrics["mae"],
    "persistence_mae": metrics["persistence_mae"],
    "test_windows": metrics["test_windows"],
    "note": "GFS-conditioned held-out replay. Metrics are normalized radar-field MAE; prototype evaluation only.",
    "observed_storms": [],
    "forecast": [],
}
for i, field in enumerate(forecast):
    q = float(np.clip(field.max(), 0, 1))
    meta["forecast"].append({
        "lead_minutes": (i + 1) * 15,
        "storms": [],
        "hazards": {
            "thunderstorm": q,
            "lightning": float(min(1, q * 1.08)),
            "hail": float(min(1, q * .72)),
            "cloudburst": float(min(1, q * .35)),
            "uncertainty": float(min(.5, .05 + i * .03)),
        },
    })

np.save(DST / "observed.npy", observed)
np.save(DST / "forecast.npy", forecast)
np.save(DST / "actual_future.npy", actual)
(DST / "metadata.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
