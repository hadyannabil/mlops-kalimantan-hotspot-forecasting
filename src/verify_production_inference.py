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

# Baca metadata Production
with open(ROOT / "deployment/production_model.yaml") as f:
    deployment = yaml.safe_load(f)

registry = deployment["registry"]
name = registry["name"]
client = MlflowClient()

# Pastikan hanya satu versi aktif di Production
versions = client.search_model_versions(
    f"name='{name}'"
)
production = [
    v for v in versions
    if v.current_stage == "Production"
]

assert len(production) == 1, (
    "Harus ada tepat satu model Production"
)

active = production[0]
assert int(active.version) == registry["active_version"]
assert active.run_id == registry["run_id"]

# Load model langsung dari Registry
model_uri = registry["model_uri"]
model = mlflow.pyfunc.load_model(model_uri)

print("=== PRODUCTION MODEL ===")
print("Model:", name)
print("Version:", active.version)
print("Stage:", active.current_stage)
print("URI:", model_uri)

# Siapkan sampel input
with open(ROOT / "model_metadata.json") as f:
    metadata = json.load(f)

features = (
    metadata["input_features"]["categorical"]
    + metadata["input_features"]["numerical"]
)

dataset = ROOT / metadata["dataset"]["file"]
df = pd.read_csv(dataset)
sample = df[features].dropna().tail(1).copy()

assert not sample.empty, "Sampel tidak tersedia"
sample["shapeID"] = sample["shapeID"].astype(str)

# Jalankan inferensi
prediction = model.predict(sample)

print("\n=== INFERENCE TEST ===")
print("Input shape:", sample.shape)
print("ShapeID:", sample["shapeID"].iloc[0])
print("Prediction:", prediction)
print("INFERENCE SUCCESS")
