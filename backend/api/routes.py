"""Product-facing prototype API."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request
from models.storm_detection import detect
from models.state_estimation import estimate

api = Blueprint("api", __name__, url_prefix="/api")
ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "data" / "demo"
GFS_CKPT = ROOT / "training" / "checkpoints_gfs_multimodal" / "best.pt"
AWS_DIR = ROOT / "data" / "raw" / "aws"


def _load_demo():
    meta = json.loads((DEMO / "metadata.json").read_text(encoding="utf-8"))
    observed = np.load(DEMO / "observed.npy")
    forecast = np.load(DEMO / "forecast.npy")
    insat = np.load(DEMO / "insat_current.npy") if (DEMO / "insat_current.npy").exists() else None
    return meta, observed, forecast, insat


def _aws_archive():
    files = sorted(AWS_DIR.glob("*.csv"))
    if not files:
        return {"available": False}
    path = files[0]
    try:
        df = pd.read_csv(path, usecols=lambda c: c in {"DATE(IST)", "TIME(IST)"})
        dt = pd.to_datetime(
            df["DATE(IST)"].astype(str) + " " + df["TIME(IST)"].astype(str),
            errors="coerce",
        ).dropna()
        return {
            "available": True,
            "type": "historical_archive",
            "station": "Shillong · ISRO0041",
            "file": path.name,
            "date_start": dt.min().strftime("%Y-%m-%d") if len(dt) else None,
            "date_end": dt.max().strftime("%Y-%m-%d") if len(dt) else None,
            "records": int(len(df)),
            "aligned_with_replay": False,
        }
    except Exception as exc:
        return {"available": True, "type": "historical_archive", "file": path.name,
                "parse_error": str(exc), "aligned_with_replay": False}


def _small(field, size=64):
    step = max(1, field.shape[0] // size)
    return field[::step, ::step][:size, :size].tolist()


def _convective_state(field, sensors):
    raw = detect(field)
    states = estimate(raw, sensors)
    intensities = [s["intensity"] for s in states]
    growth = [s["state"]["growth_proxy"] for s in states]
    speeds = [float(np.hypot(*s["state"]["motion"])) for s in states]
    return {
        "cells": len(states),
        "mean_intensity": float(np.mean(intensities)) if intensities else 0.0,
        "max_intensity": float(max(intensities)) if intensities else float(np.max(field)),
        "mean_motion": float(np.mean(speeds)) if speeds else 0.0,
        "growth_proxy": float(np.mean(growth)) if growth else 0.0,
        "state_quality": float(np.mean([s["state"]["quality"] for s in states])) if states else 0.0,
        "field_energy": float(np.mean(field)),
        "convective_fraction": float(np.mean(field >= 0.15)),
        "strong_fraction": float(np.mean(field >= 0.60)),
    }


@api.get("/run")
def run_pipeline():
    return state()


@api.get("/state")
def state():
    if not (DEMO / "metadata.json").exists():
        return jsonify({"mode": "no_data", "field": None, "objects": [],
                        "sensors": {}, "sensor_details": {}, "hazards": {}, "quality": None,
                        "convective_state": {}})

    meta, observed, forecast, insat = _load_demo()
    lead = int(request.args.get("t", 0))
    gfs_ready = GFS_CKPT.exists()
    aws = _aws_archive()
    # AWS is deliberately NOT marked as a live model input: its Shillong archive
    # is 2018-12-03..2019-01-03 while this replay is a 2026 DWR/INSAT event.
    sensors = {"DWR": 1.0, "INSAT": 1.0 if insat is not None else 0.0,
               "ILDN": 0.0, "AWS": 0.0, "GFS": 1.0 if gfs_ready else 0.0}

    if lead <= 0:
        field = observed
        objects = detect(field)
        if meta.get("observed_storms"):
            objects = meta["observed_storms"]
        hazards = {}
    else:
        idx = min(max(lead // 15 - 1, 0), len(forecast) - 1)
        item = meta["forecast"][idx]
        field = forecast[idx]
        objects = item["storms"]
        hazards = item["hazards"]

    convective_state = _convective_state(field, sensors)
    if objects:
        estimated = estimate([dict(o) for o in objects], sensors)
        convective_state["cells"] = len(estimated)
        convective_state["mean_intensity"] = float(np.mean([o["state"]["intensity"] for o in estimated]))
        convective_state["growth_proxy"] = float(np.mean([o["state"]["growth_proxy"] for o in estimated]))
        convective_state["mean_motion"] = float(np.mean([np.hypot(*o["state"]["motion"]) for o in estimated]))

    return jsonify({
        "mode": "prototype_replay", "field": _small(field), "objects": objects,
        "sensors": sensors, "sensor_details": {
            "DWR": {"status": "live_replay_input", "label": "Available"},
            "INSAT": {"status": "replay_input" if insat is not None else "unavailable", "label": "Available" if insat is not None else "Not available"},
            "ILDN": {"status": "unavailable", "label": "Not available"},
            "AWS": {**aws, "label": "Historical archive" if aws.get("available") else "Not available"},
            "GFS": {"status": "model_input" if gfs_ready else "unavailable", "label": "Available" if gfs_ready else "Not available"},
        },
        "hazards": hazards, "convective_state": convective_state,
        "quality": float(np.mean([v for k, v in sensors.items() if v > 0])),
        "lead_minutes": lead, "source_event": meta["source_event"],
        "model": "DWR + INSAT-3DR CTP/CTT + GFS ConvLSTM" if gfs_ready else meta["model"],
        "metrics": {"heldout_mae": meta.get("heldout_mae"),
                    "persistence_mae": meta.get("persistence_mae"),
                    "checkpoint_epoch": meta.get("checkpoint_epoch"),
                    "gfs_checkpoint_ready": gfs_ready},
    })


@api.get("/health")
def health():
    gfs_ready = GFS_CKPT.exists()
    return jsonify({"status": "ok", "project": "VAJRA-CSDD",
                    "mode": "prototype_replay" if (DEMO / "metadata.json").exists() else "no_data",
                    "pipeline": "ready", "gfs_model": "ready" if gfs_ready else "training"})
