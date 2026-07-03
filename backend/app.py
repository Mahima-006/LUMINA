import os, json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import tensorflow as tf
from predictor import MEAN_FEATURE_NAMES, VALID_RANGES, predict

app = Flask(__name__)

allowed_origin = os.environ.get("ALLOWED_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": allowed_origin}})

BASE = os.path.dirname(__file__)

# Load all 4 artifacts at startup — fail loudly if missing
model  = tf.keras.models.load_model(
           os.path.join(BASE, "breast_cancer_nn.keras"))
scaler = load(os.path.join(BASE, "scaler.pkl"))

with open(os.path.join(BASE, "feature_names.json")) as f:
  all_feature_names = json.load(f)
with open(os.path.join(BASE, "se_worst_means.json")) as f:
  se_worst_means = json.load(f)

@app.route("/health", methods=["GET"])
def health():
  return jsonify({"status": "ok", "model": "breast_cancer_nn"})

@app.route("/features", methods=["GET"])
def features():
  return jsonify({
    "features": [
      {
        "name": feat,
        "min": VALID_RANGES[feat][0],
        "max": VALID_RANGES[feat][1],
      }
      for feat in MEAN_FEATURE_NAMES
    ]
  })

@app.route("/predict", methods=["POST"])
def predict_route():
  data   = request.get_json(force=True)
  inputs = data.get("inputs", {})
  FIELD_MAPPING = {
    "radius": "mean radius",
    "texture": "mean texture",
    "perimeter": "mean perimeter",
    "area": "mean area",
    "smoothness": "mean smoothness",
    "compactness": "mean compactness",
    "concavity": "mean concavity",
    "concave_points": "mean concave points",
    "symmetry": "mean symmetry",
    "fractal_dim": "mean fractal dimension",
}

  mapped_inputs = {}

  for key, value in inputs.items():
      mapped_inputs[FIELD_MAPPING.get(key, key)] = value

  inputs = mapped_inputs

  errors = []

  for feat in MEAN_FEATURE_NAMES:
    if feat not in inputs:
      errors.append(f"'{feat}' is required.")
      continue
    try:
      val = float(inputs[feat])
    except (TypeError, ValueError):
      errors.append(f"'{feat}' must be a number.")
      continue
    lo, hi = VALID_RANGES[feat]
    if not (lo <= val <= hi):
      errors.append(f"'{feat}' must be between {lo} and {hi}.")

  if errors:
    return jsonify({"error": " | ".join(errors)}), 400

  result = predict(inputs, model, scaler,
                   all_feature_names, se_worst_means)
  return jsonify({
    "result": result["result"],
    "confidence": result["confidence"],
    "probabilities": {
        "benign": result["benign_probability"],
        "malignant": result["malignant_probability"]
    },
    "disclaimer": result["disclaimer"]
})

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
