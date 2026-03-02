"""
inference_server.py
───────────────────
Persistent Flask HTTP wrapper around risk_inference.py.
Run once alongside your Node.js server — loads the model into memory once,
then serves predictions over HTTP with no per-request startup cost.

Start:
    python inference_server.py

Node.js calls:
    POST http://localhost:5001/score        { ...feat_dict }
    POST http://localhost:5001/score/batch  [ {...}, {...} ]
    GET  http://localhost:5001/health
"""

from flask import Flask, request, jsonify
from infer import score, score_batch, _load_artifacts, predict_accident
import os

app = Flask(__name__)

# Load model/scaler/baseline into memory at startup — not on first request
print("Loading model artifacts...")
_load_artifacts()
print("Ready.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/score", methods=["POST"])
def score_single():
    feat = request.get_json(force=True)
    if not isinstance(feat, dict):
        return jsonify({"error": "Expected a JSON object"}), 400
    try:
        return jsonify(score(feat))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/score/batch", methods=["POST"])
def score_batch_route():
    feats = request.get_json(force=True)
    if not isinstance(feats, list):
        return jsonify({"error": "Expected a JSON array"}), 400
    try:
        return jsonify(score_batch(feats))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/score/image", methods=["POST"])
def score_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    try:
        image_bytes = file.read()
        result = predict_accident(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"CNN inference failed: {str(e)}"}), 500
    

if __name__ == "__main__":
       port = int(os.environ.get("PORT", 5001))
       app.run(host="0.0.0.0", port=port, threaded=False)