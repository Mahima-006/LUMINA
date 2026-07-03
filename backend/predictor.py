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
import pandas as pd

def build_full_input_vector(user_inputs, se_worst_means,
                            all_feature_names, scaler):

    print("BUILD 1")

    mean_values = np.array([
        float(user_inputs[feat])
        for feat in MEAN_FEATURE_NAMES
    ])

    print("BUILD 2")

    full_raw = np.concatenate([
        mean_values,
        scaler.mean_[10:]
    ])

    print("BUILD 3")
    print(full_raw.shape)

    df = pd.DataFrame([full_raw], columns=all_feature_names)

    print("BUILD 4")

    X = scaler.transform(df)

    print("BUILD 5")

    return X
def predict(user_inputs, model, scaler,
            all_feature_names,
            se_worst_means):

    print("STEP 1")
    X = build_full_input_vector(
        user_inputs,
        se_worst_means,
        all_feature_names,
        scaler,
    )

    print("STEP 2")
    print(X)

    prediction_proba = model.predict(X, verbose=0)

    print("STEP 3")
    print(prediction_proba)

    pred_label = int(np.argmax(prediction_proba[0]))

    print("STEP 4")

    benign_prob = round(float(prediction_proba[0][1]) * 100, 1)
    malignant_prob = round(float(prediction_proba[0][0]) * 100, 1)
    confidence = round(float(np.max(prediction_proba[0])) * 100, 1)

    result_label = "Benign" if pred_label == 1 else "Malignant"
    print("X =", X)
    print("Shape =", X.shape)

    prediction_proba = model.predict(X, verbose=0)

    print("Prediction =", prediction_proba)
    return {
        "result": result_label,
        "confidence": confidence,
        "benign_probability": benign_prob,
        "malignant_probability": malignant_prob,
        "disclaimer":
            "This is an educational ML demonstration and does NOT constitute medical advice."
    }
    
