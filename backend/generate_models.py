import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from joblib import dump
import json
import os

# Create base dir if running elsewhere, we want files in this same directory
BASE = os.path.dirname(__file__)

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

data_frame = pd.DataFrame(
  breast_cancer_dataset.data,
  columns=breast_cancer_dataset.feature_names
)
data_frame['label'] = breast_cancer_dataset.target

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

X_train, X_test, Y_train, Y_test = train_test_split(
  X, Y, test_size=0.2, random_state=2
)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)

model = keras.Sequential([
  keras.layers.Flatten(input_shape=(30,)),
  keras.layers.Dense(20, activation='relu'),
  keras.layers.Dense(2, activation='softmax')
])

model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

history = model.fit(
  X_train_std, Y_train,
  validation_split=0.1,
  epochs=10,
  verbose=1
)

loss, accuracy = model.evaluate(X_test_std, Y_test, verbose=1)
print(f"Test Accuracy: {accuracy:.4f}")

# --- 11. EXPORT ---
# Save Keras model
model.save(os.path.join(BASE, 'breast_cancer_nn.keras'))

# Save scaler
dump(scaler, os.path.join(BASE, 'scaler.pkl'))

# Save feature names
feature_names = list(breast_cancer_dataset.feature_names)
with open(os.path.join(BASE, 'feature_names.json'), 'w') as f:
  json.dump(feature_names, f)

# Save training set means for the 20 non-user-input features
X_train_df = pd.DataFrame(X_train_std, columns=breast_cancer_dataset.feature_names)
se_worst_means = {
  col: float(X_train_df[col].mean())
  for col in breast_cancer_dataset.feature_names[10:]
}
with open(os.path.join(BASE, 'se_worst_means.json'), 'w') as f:
  json.dump(se_worst_means, f)

print("Exported: breast_cancer_nn.keras, scaler.pkl, feature_names.json, se_worst_means.json")
