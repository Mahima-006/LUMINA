import numpy as np
import json

# The 10 mean feature names (indices 0–9 in sklearn order)
# These are the fields the user fills in via the form.
MEAN_FEATURE_NAMES = [
  'mean radius', 'mean texture', 'mean perimeter', 'mean area',
  'mean smoothness', 'mean compactness', 'mean concavity',
  'mean concave points', 'mean symmetry', 'mean fractal dimension'
]

# Valid input ranges (based on dataset min/max with small buffer)
VALID_RANGES = {
  'mean radius':           (6.0,   30.0),
  'mean texture':          (9.0,   40.0),
  'mean perimeter':        (43.0,  195.0),
  'mean area':             (140.0, 2510.0),
  'mean smoothness':       (0.05,  0.17),
  'mean compactness':      (0.01,  0.36),
  'mean concavity':        (0.0,   0.44),
  'mean concave points':   (0.0,   0.21),
  'mean symmetry':         (0.10,  0.32),
  'mean fractal dimension':(0.04,  0.10),
}

def build_full_input_vector(user_inputs: dict,
                             se_worst_means: dict,
                             all_feature_names: list,
                             scaler) -> np.ndarray:
  """
  Takes 10 mean feature values from the user form.
  Fills indices 10–29 (SE and worst features) with their 
  standardized training-set mean values (effectively 0 after 
  standardization — these are pre-standardized means).
  Returns a scaled (1, 30) numpy array for model.predict().
  """
  # Build raw (un-scaled) row for mean features only
  # The scaler was fit on all 30 — we need to invert for the SE/worst
  # Strategy: pass mean features through scaler, fill SE/worst 
  # with their post-standardization mean (0.0) directly.
  
  mean_values = np.array([
    float(user_inputs[feat]) for feat in MEAN_FEATURE_NAMES
  ]).reshape(1, -1)

  # Scale the 10 mean features using the full 30-feature scaler
  # by temporarily building a 30-feature row where SE/worst = 
  # their un-scaled training means (scaler.mean_[10:])
  full_raw = np.concatenate([
    mean_values[0],
    scaler.mean_[10:]   # un-scaled means for features 10–29
  ]).reshape(1, -1)

  return scaler.transform(full_raw)

def predict(user_inputs: dict, model, scaler,
            all_feature_names: list,
            se_worst_means: dict) -> dict:

  X = build_full_input_vector(
    user_inputs, se_worst_means, all_feature_names, scaler
  )
  prediction_proba = model.predict(X, verbose=0)   # shape (1, 2)
  pred_label = int(np.argmax(prediction_proba[0]))

  benign_prob    = round(float(prediction_proba[0][1]) * 100, 1)
  malignant_prob = round(float(prediction_proba[0][0]) * 100, 1)
  confidence     = round(float(np.max(prediction_proba[0])) * 100, 1)
  result_label   = "Benign" if pred_label == 1 else "Malignant"

  return {
    "result": result_label,
    "confidence": confidence,
    "benign_probability": benign_prob,
    "malignant_probability": malignant_prob,
    "disclaimer": (
      "This is an educational ML demonstration and does NOT "
      "constitute medical advice. Consult a licensed medical "
      "professional for any health concerns."
    )
  }
