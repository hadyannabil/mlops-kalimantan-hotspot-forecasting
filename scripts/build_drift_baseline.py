import bisect
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/hotspot_model_ready.csv"
META_PATH = ROOT / "data/processed/model_data_metadata.json"

OUTPUT_PATH = (
    ROOT / "deployment/serving/drift_baseline.json"
)

# Fixed histogram boundaries for each monitored feature
FEATURE_BINS = {
    "hotspot_count": [0, 1, 3, 7, 15, 31, 63],
    "delta": [-16, -8, -3, -1, 0, 1, 3, 8, 16, 32],
    "rolling_mean_7": [0, 0.5, 1, 2, 4, 8, 16, 32],
}

metadata = json.loads(META_PATH.read_text())

period_start, period_end = metadata["periods"]["training"]

counts = {
    feature: [0] * (len(bins) + 1)
    for feature, bins in FEATURE_BINS.items()
}

dates = []
total = 0

with DATA_PATH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["split"] != "training":
            continue

        if not period_start <= row["date"] <= period_end:
            raise ValueError("Training date outside expected period")

        for feature, bins in FEATURE_BINS.items():
            value = float(row[feature])

            if not math.isfinite(value):
                raise ValueError(f"Invalid value: {feature}")

            index = bisect.bisect_left(bins, value)
            counts[feature][index] += 1

        dates.append(row["date"])
        total += 1

if total == 0:
    raise ValueError("Training dataset is empty")

baseline = {
    "reference_split": "training",
    "dataset_sha256": hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest(),
    "training_rows": total,
    "date_start": min(dates),
    "date_end": max(dates),
    "features": {}
}

for feature, bins in FEATURE_BINS.items():
    proportions = [
        count / total for count in counts[feature]
    ]

    baseline["features"][feature] = {
        "upper_bounds": bins + ["+Inf"],
        "counts": counts[feature],
        "proportions": proportions,
    }

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH.write_text(
    json.dumps(baseline, indent=2),
    encoding="utf-8"
)

print("=== DRIFT BASELINE GENERATED ===")
print("Training rows:", total)
print("Actual period:", min(dates), "to", max(dates))
print("Dataset SHA256:", baseline["dataset_sha256"])
print("Output:", OUTPUT_PATH)

for feature, info in baseline["features"].items():
    print(f"\n=== {feature} ===")

    for bound, count, prop in zip(
        info["upper_bounds"],
        info["counts"],
        info["proportions"],
    ):
        print(
            f"Upper bound: {str(bound):>5} | "
            f"Count: {count:>6} | "
            f"Proportion: {prop:.4%}"
        )
