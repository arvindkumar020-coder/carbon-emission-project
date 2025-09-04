from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import json

# --------------------
# Paths
# --------------------
DATA_PATH = "data/vehicles_100_corrected.csv"   # location of dataset
MODEL_PATH = "ml/model.pkl"
META_PATH = "ml/metadata.json"

# --------------------
# Define schema
# --------------------
CATEGORICAL_COLS = ["Make", "Model", "Fuel"]
NUMERIC_COLS = ["EngineSize", "Cylinders", "FuelConsumption"]
TARGET = "CO2Emissions"
REQUIRED_COLUMNS = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET]

# --------------------
# Load data
# --------------------
print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Ensure required columns exist
missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    raise ValueError(f"Dataset missing required columns: {missing_cols}")

X = df[CATEGORICAL_COLS + NUMERIC_COLS]
y = df[TARGET]

# --------------------
# Preprocessing
# --------------------
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, CATEGORICAL_COLS),
        ("num", numeric_transformer, NUMERIC_COLS),
    ]
)

# --------------------
# Model
# --------------------
reg = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=None,
    n_jobs=-1,
)

pipe = Pipeline(steps=[("prep", preprocessor), ("rf", reg)])

# --------------------
# Train/test split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe.fit(X_train, y_train)

# --------------------
# Evaluation
# --------------------
pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
print({"MAE_g_per_km": round(mae, 2), "RMSE_g_per_km": round(rmse, 2)})

# --------------------
# Save artifacts
# --------------------
joblib.dump(pipe, MODEL_PATH)

feature_info = {
    "categorical": CATEGORICAL_COLS,
    "numeric": NUMERIC_COLS,
    "target": TARGET,
    "metrics": {"MAE_g_per_km": float(mae), "RMSE_g_per_km": float(rmse)},
}

with open(META_PATH, "w") as f:
    json.dump(feature_info, f, indent=4)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Metadata saved to {META_PATH}")
