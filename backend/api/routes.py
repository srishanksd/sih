from flask import Blueprint, jsonify, request

api = Blueprint("api", __name__, url_prefix="/api")


@api.get("/run")
def run_pipeline():
    # Real-data pipeline will be connected here once source adapters are added.
    _ = request.args.get("t", 0)
    _ = request.args.get("lead", 60)
    return jsonify({
        "mode": "no_data",
        "field": None,
        "objects": [],
        "sensors": {},
        "hazards": {},
        "quality": None,
        "lead_minutes": None,
    })
