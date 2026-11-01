import json
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri(
    f"sqlite:///{ROOT / 'mlflow.db'}"
)

with open(ROOT / "deployment/production_model.yaml") as f:
    deployment = yaml.safe_load(f)

with open(ROOT / "model_metadata.json") as f:
    metadata = json.load(f)

registry = deployment["registry"]
client = MlflowClient()

active = client.get_model_version(
    registry["name"],
    str(registry["active_version"])
)
assert active.current_stage == "Production"

model = mlflow.pyfunc.load_model(
    registry["model_uri"]
)

# Wilayah yang sudah dikenal model
shape_id = "22746128B97983454808676"
date = pd.Timestamp("2026-10-01")

# Delapan hari berurutan: t-7 sampai t
scenarios = {
    "Aktivitas rendah": [0, 1, 0, 0, 1, 1, 2, 1],
    "Aktivitas meningkat": [2, 3, 2, 4, 5, 6, 8, 22],
}

def build_features(counts):
    last7 = pd.Series(counts[-7:], dtype=float)

    return {
        "shapeID": shape_id,
        "hotspot_count": counts[-1],
        "delta": counts[-1] - counts[-2],
        "lag_1": counts[-2],
        "lag_2": counts[-3],
        "lag_3": counts[-4],
        "lag_7": counts[-8],
        "rolling_mean_3": sum(counts[-3:]) / 3,
        "rolling_mean_7": last7.mean(),
        "rolling_max_7": last7.max(),
        "rolling_std_7": last7.std(),
        "month": date.month,
        "day_of_year": date.dayofyear,
    }

features = (
    metadata["input_features"]["categorical"]
    + metadata["input_features"]["numerical"]
)

inputs = pd.DataFrame([
    build_features(counts)
    for counts in scenarios.values()
])[features]

# Samakan tipe data dengan MLflow Model Signature
inputs["shapeID"] = inputs["shapeID"].astype("str")

integer_features = [
    "hotspot_count",
    "month",
    "day_of_year",
]

for col in features:
    if col == "shapeID":
        continue
    elif col in integer_features:
        inputs[col] = inputs[col].astype("int64")
    else:
        inputs[col] = inputs[col].astype("float64")
        
assert inputs.shape == (2, 13)
assert inputs.notna().all().all()

predictions = model.predict(inputs)
assert len(predictions) == 2

print("=== NEW INPUT INFERENCE ===")
print("Model:", registry["name"])
print("Version:", active.version)
print("Stage:", active.current_stage)
print("Date:", date.date())
print("Input shape:", inputs.shape)

for name, pred in zip(scenarios, predictions):
    label = int(pred)
    assert label in (0, 1)
    status = "SPIKE" if label == 1 else "NO SPIKE"
    print(f"{name}: {label} ({status})")

print("NEW INPUT INFERENCE SUCCESS")
