from __future__ import annotations
from pathlib import Path
from typing import Dict, Any


import numpy as np
import pandas as pd


CATEGORICAL_COLS = [
"make", "model", "vehicle_class", "transmission", "fuel_type"
]
NUMERIC_COLS = [
"year", "engine_displ_l", "cylinders",
"fuel_cons_l_per_100km_city", "fuel_cons_l_per_100km_hwy",
"curb_weight_kg", "power_kw", "stop_start"
]
TARGET = "co2_g_km"


REQUIRED_COLUMNS = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET]




def clean_df(df: pd.DataFrame) -> pd.DataFrame:
df = df.copy()
# Standardize text
for c in ["make", "model", "vehicle_class", "transmission", "fuel_type"]:
if c in df:
df[c] = (
df[c]
.astype(str)
.str.strip()
.str.replace("\\s+", " ", regex=True)
.str.title()
)
# Ensure numeric types
numeric_types = {
"year": int,
"engine_displ_l": float,
"cylinders": int,
"fuel_cons_l_per_100km_city": float,
"fuel_cons_l_per_100km_hwy": float,
"curb_weight_kg": float,
"power_kw": float,
"stop_start": int,
TARGET: float,
}
for col, typ in numeric_types.items():
if col in df:
df[col] = pd.to_numeric(df[col], errors="coerce").astype(typ, errors="ignore")
# Drop impossible rows
df = df.dropna(subset=[TARGET])
df = df[(df["engine_displ_l"] > 0) & (df["curb_weight_kg"] > 0)]
# Replace missing with medians/modes later in Pipeline
return df




def save_metadata(path: Path, meta: Dict[str, Any]):
path.write_text(json.dumps(meta, indent=2))