import argparse
import json
import time
import urllib.request

from concurrent.futures import ThreadPoolExecutor
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--duration", type=int, default=120)
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

if args.duration <= 0 or args.workers <= 0:
    parser.error("Duration and workers must be positive")

URL = "http://127.0.0.1:8002/invocations"

payload = {
    "dataframe_records": [{
        "shapeID": "TEST",
        "hotspot_count": 10,
        "delta": 2.0,
        "lag_1": 8.0,
        "lag_2": 7.0,
        "lag_3": 6.0,
        "lag_7": 5.0,
        "rolling_mean_3": 7.0,
        "rolling_mean_7": 6.0,
        "rolling_max_7": 10.0,
        "rolling_std_7": 2.0,
        "month": 11,
        "day_of_year": 331
    }]
}

body = json.dumps(payload).encode()

def worker(deadline):
    results = Counter()

    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                URL,
                data=body,
                headers={
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            with urllib.request.urlopen(
                req, timeout=30
            ) as response:

                output = json.loads(response.read())

                if len(output["predictions"]) != 1:
                    raise ValueError("Invalid prediction count")

                results["success"] += 1

                backend = response.headers.get(
                    "X-Backend", "unknown"
                )

                results[f"backend:{backend}"] += 1

        except Exception:
            results["failed"] += 1

        time.sleep(0.2)

    return results

print("=== LK11 FINAL LOAD TEST ===", flush=True)
print("Duration:", args.duration, "seconds", flush=True)
print("Concurrent workers:", args.workers, flush=True)

total = Counter()
start = time.monotonic()
deadline = start + args.duration

with ThreadPoolExecutor(
    max_workers=args.workers
) as executor:

    futures = [
        executor.submit(worker, deadline)
        for _ in range(args.workers)
    ]

    for future in futures:
        total.update(future.result())

elapsed = time.monotonic() - start

success = total["success"]
failed = total["failed"]
requests = success + failed

print("\n=== LOAD TEST RESULTS ===")
print("Successful requests:", success)
print("Failed requests:", failed)
print("Total requests:", requests)
print(f"Actual duration: {elapsed:.2f} seconds")
print(f"Average throughput: {requests / elapsed:.2f} req/s")

if requests:
    print(f"Success rate: {success / requests:.2%}")

print("\nBackend distribution:")

for key, value in sorted(total.items()):
    if key.startswith("backend:"):
        print(key.removeprefix("backend:"), ":", value)
