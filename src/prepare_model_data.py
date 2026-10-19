import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/daily_hotspot_activity.csv"
OUTPUT = ROOT / "data/processed/hotspot_model_ready.csv"
METADATA = ROOT / "data/processed/model_data_metadata.json"

df = pd.read_csv(SOURCE, dtype={"shapeID": "string"})
df["date"] = pd.to_datetime(df["date"])

df = df[df["date"].between(
    "2025-01-01", "2026-09-30"
)].copy()

df = df.sort_values(["shapeID", "date"])

# Validasi grid harian
expected = pd.date_range("2025-01-01", "2026-09-30")
assert df["date"].nunique() == len(expected)
assert df.groupby("date").size().eq(56).all()
assert not df.duplicated(["date", "shapeID"]).any()
assert df["hotspot_count"].notna().all()

# Feature engineering menggunakan informasi sampai t
g = df.groupby("shapeID")

df["delta"] = g["hotspot_count"].diff()

for lag in [1, 2, 3, 7]:
    df[f"lag_{lag}"] = g["hotspot_count"].shift(lag)

for window in [3, 7]:
    df[f"rolling_mean_{window}"] = (
        g["hotspot_count"].transform(
            lambda x: x.rolling(window).mean()
        )
    )

df["rolling_max_7"] = g["hotspot_count"].transform(
    lambda x: x.rolling(7).max()
)
df["rolling_std_7"] = g["hotspot_count"].transform(
    lambda x: x.rolling(7).std()
)

df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear

features = [
    "hotspot_count", "delta",
    "lag_1", "lag_2", "lag_3", "lag_7",
    "rolling_mean_3", "rolling_mean_7",
    "rolling_max_7", "rolling_std_7",
    "month", "day_of_year",
]

# Threshold hanya dari training
train_mask = df["date"] <= "2026-06-30"
positive = df.loc[
    train_mask & (df["delta"] > 0), "delta"
]
threshold = float(positive.quantile(0.90))

# Target: minimal satu lonjakan pada t+1, t+2, t+3
g = df.groupby("shapeID")

for h in [1, 2, 3]:
    df[f"future_delta_{h}"] = g["delta"].shift(-h)
    df[f"future_date_{h}"] = g["date"].shift(-h)

future_cols = [f"future_delta_{h}" for h in [1, 2, 3]]

df["future_max_delta"] = df[future_cols].max(axis=1)

periods = {
    "training": ("2025-01-01", "2026-06-30"),
    "validation": ("2026-07-01", "2026-07-31"),
    "test": ("2026-08-01", "2026-09-30"),
}

parts = []

for name, (start, end) in periods.items():
    last_reference = pd.Timestamp(end) - pd.Timedelta(days=3)

    valid = df["date"].between(start, last_reference)

    for h in [1, 2, 3]:
        valid &= (
            df[f"future_date_{h}"]
            == df["date"] + pd.Timedelta(days=h)
        )

    part = df.loc[valid].dropna(
        subset=features + future_cols
    ).copy()

    part["spike"] = (
        part["future_max_delta"] > threshold
    ).astype(int)

    part["split"] = name
    parts.append(part)

result = pd.concat(parts, ignore_index=True)

columns = [
    "date", "shapeID", "province",
    "kabupaten_kota", *features, "spike", "split"
]

result[columns].to_csv(OUTPUT, index=False)

metadata = {
    "source": str(SOURCE.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest(),
    "target": "SPIKE within t+1 to t+3",
    "threshold_method": "P90 positive training deltas",
    "threshold_spike": threshold,
    "feature_columns": features,
    "categorical_region_column": "shapeID",
    "periods": periods,
}
METADATA.write_text(json.dumps(metadata, indent=2))

print("=== MODEL-READY DATASET ===")
print("Threshold SPIKE:", threshold)
print("Jumlah fitur numerik:", len(features))

for name, part in result.groupby("split"):
    n_spike = int(part["spike"].sum())
    print(
        f"{name}: {len(part):,} baris, "
        f"SPIKE={n_spike:,}, "
        f"SPIKE%={part['spike'].mean()*100:.2f}%"
    )

print("Dataset:", OUTPUT)
print("Metadata:", METADATA)
