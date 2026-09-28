"""Product-facing prototype API."""
from pathlib import Path
import json
import numpy as np
from flask import Blueprint, jsonify, request

api = Blueprint("api", __name__, url_prefix="/api")
ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "demo"


def _load_demo():
    meta = json.loads((DEMO / "metadata.json").read_text(encoding="utf-8"))
    observed = np.load(DEMO / "observed.npy")
    forecast = np.load(DEMO / "forecast.npy")
    return meta, observed, forecast


def _small(field, size=64):
    step = max(1, field.shape[0] // size)
    return field[::step, ::step][:size, :size].tolist()


@api.get("/run")
def run_pipeline():
    return state()


@api.get("/state")
def state():
    if not (DEMO / "metadata.json").exists():
        return jsonify({
            "mode": "no_data", "field": None, "objects": [],
            "sensors": {}, "hazards": {}, "quality": None,
            "lead_minutes": None,
        })

    meta, observed, forecast = _load_demo()
    lead = int(request.args.get("t", 0))
    if lead <= 0:
        field = observed
        objects = meta["observed_storms"]
        hazards = {}
        mode = "prototype_replay"
    else:
        idx = min(max(lead // 15 - 1, 0), len(forecast) - 1)
        item = meta["forecast"][idx]
        field = forecast[idx]
        objects = item["storms"]
        hazards = item["hazards"]
        mode = "prototype_replay"

    return jsonify({
        "mode": mode,
        "field": _small(field),
        "objects": objects,
        "sensors": {
            "DWR": 1.0,
            "INSAT": None,
            "ILDN": None,
            "AWS": None,
            "GFS": None,
        },
        "hazards": hazards,
        "quality": 0.88,
        "lead_minutes": lead,
        "source_event": meta["source_event"],
        "model": meta["model"],
    })


@api.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "project": "VAJRA-CSDD",
        "mode": "prototype_replay" if (DEMO / "metadata.json").exists() else "no_data",
        "pipeline": "ready",
    })
