import time

import mlflow
import uvicorn
from fastapi import Request, Response
from mlflow.pyfunc import scoring_server
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

MODEL_PATH = "/opt/ml/model"

from drift_metrics import MonitoredModel

model = mlflow.pyfunc.load_model(MODEL_PATH)
model = MonitoredModel(model, MODEL_PATH)
app = scoring_server.init(model)

# Jumlah request inferensi berdasarkan status HTTP
INFERENCE_REQUESTS = Counter(
    "model_inference_requests_total",
    "Total inference requests",
    ["status_code"],
)

# Durasi pemrosesan request inferensi dalam detik
INFERENCE_LATENCY = Histogram(
    "model_inference_duration_seconds",
    "Inference request duration in seconds",
    buckets=(
        0.005, 0.01, 0.025, 0.05,
        0.1, 0.25, 0.5, 1.0,
        2.5, 5.0, 10.0, 30.0
    ),
)


@app.middleware("http")
async def monitor_inference(request: Request, call_next):
    if request.url.path != "/invocations":
        return await call_next(request)

    start = time.perf_counter()
    status_code = "500"

    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        return response

    finally:
        INFERENCE_REQUESTS.labels(
            status_code=status_code
        ).inc()

        INFERENCE_LATENCY.observe(
            time.perf_counter() - start
        )


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        workers=1,
    )
