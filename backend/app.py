from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from api.routes import api

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)
app.register_blueprint(api)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "project": "VAJRA-CSDD",
        "mode": "no_data",
        "pipeline": "ready",
    })


@app.get("/api/state")
def state():
    return jsonify({
        "mode": "no_data",
        "field": None,
        "objects": [],
        "sensors": {},
        "hazards": {},
        "quality": None,
        "lead_minutes": None,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
