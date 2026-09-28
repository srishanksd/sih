from flask import Blueprint, jsonify, request

from core.pipeline import run as run_nowcasting_pipeline

api = Blueprint("api", __name__, url_prefix="/api")


@api.get("/run")
def run_pipeline():
    # The API is the product-facing entry point into the nowcasting core.
    # Real observations will be supplied by source adapters when connected.
    _ = request.args.get("t", 0)
    _ = request.args.get("lead", 60)
    _ = run_nowcasting_pipeline

    return jsonify({
        "mode": "no_data",
        "field": None,
        "objects": [],
        "sensors": {},
        "hazards": {},
        "quality": None,
        "lead_minutes": None,
    })
