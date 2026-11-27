import argparse
import csv
import json
import random
import time
import urllib.request

from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/hotspot_model_ready.csv"
URL = "http://127.0.0.1:8002/invocations"

FEATURES = [
    "shapeID",
    "hotspot_count",
    "delta",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_max_7",
    "rolling_std_7",
    "month",
    "day_of_year",
]

INTEGER_FEATURES = {
    "hotspot_count",
    "month",
    "day_of_year",
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["normal", "drift"],
    required=True
)
parser.add_argument(
    "--duration",
    type=int,
    default=160
)

args = parser.parse_args()

rng = random.Random(42)

rows = []

with DATA_PATH.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):

        if row["split"] != "training":
            continue

        if args.mode == "drift":
            if not (
                float(row["hotspot_count"]) >= 8
                and float(row["delta"]) >= 1
                and float(row["rolling_mean_7"]) >= 4
            ):
                continue

        record = {}

        for feature in FEATURES:
            if feature == "shapeID":
                record[feature] = row[feature]

            elif feature in INTEGER_FEATURES:
                record[feature] = int(float(row[feature]))

            else:
                record[feature] = float(row[feature])

        rows.append(record)

if not rows:
    raise RuntimeError("No eligible observations")

BATCH_SIZE = 20
results = Counter()
predictions = Counter()
backends = Counter()

print("=== DRIFT SIMULATION ===", flush=True)
print("Mode:", args.mode, flush=True)
print("Available observations:", len(rows), flush=True)
print("Batch size:", BATCH_SIZE, flush=True)
print("Duration:", args.duration, "seconds", flush=True)

start = time.monotonic()
last_report = 0

while time.monotonic() - start < args.duration:

    batch = rng.choices(rows, k=BATCH_SIZE)

    payload = json.dumps({
        "dataframe_records": batch
    }).encode()

    try:
        req = urllib.request.Request(
            URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=15) as res:

            response = json.loads(res.read())
            output = response["predictions"]

            if len(output) != BATCH_SIZE:
                raise ValueError("Prediction count mismatch")

            results["success"] += 1
            results["observations"] += len(output)

            backends[
                res.headers.get("X-Backend", "unknown")
            ] += 1

            for value in output:
                label = "SPIKE" if int(value) == 1 else "NO_SPIKE"
                predictions[label] += 1

    except Exception as exc:
        results["failed"] += 1

        if results["failed"] <= 3:
            print("Error:", exc, flush=True)

    elapsed = int(time.monotonic() - start)

    if elapsed - last_report >= 10:
        print(
            f"[{elapsed}s] "
            f"Requests: {results['success']} | "
            f"Observations: {results['observations']} | "
            f"Failed: {results['failed']}",
            flush=True
        )
        last_report = elapsed

    time.sleep(1)

print("\n=== SIMULATION RESULTS ===")
print("Mode:", args.mode)
print("Successful requests:", results["success"])
print("Failed requests:", results["failed"])
print("Total observations:", results["observations"])

print("\nPrediction distribution:")
for key, value in predictions.items():
    print(key, ":", value)

print("\nBackend distribution:")
for key, value in backends.items():
    print(key, ":", value)
