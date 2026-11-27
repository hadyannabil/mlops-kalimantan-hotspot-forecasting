import bisect
import json
import logging
import math
from pathlib import Path

import mlflow.sklearn
import numpy as np
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

BASELINE_PATH = Path(__file__).with_name(
    "drift_baseline.json"
)

BASELINE = json.loads(BASELINE_PATH.read_text())

PREDICTIONS = Counter(
    "model_predictions_total",
    "Number of predicted observations",
    ["prediction"],
)

PROBABILITY = Histogram(
    "model_spike_probability",
    "Predicted probability of SPIKE",
    buckets=(
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0
    ),
)

FEATURE_BINS = Counter(
    "model_input_feature_bin_total",
    "Incoming observations per feature interval",
    ["feature", "bin"],
)

REFERENCE = Gauge(
    "model_input_feature_reference_ratio",
    "Training distribution per feature interval",
    ["feature", "bin"],
)

MONITORING_ERRORS = Counter(
    "model_monitoring_errors_total",
    "Errors while collecting prediction metrics",
)

# Initialize metrics and reference distributions
for prediction in ["NO_SPIKE", "SPIKE"]:
    PREDICTIONS.labels(prediction=prediction)

for feature, info in BASELINE["features"].items():
    for index, proportion in enumerate(
        info["proportions"]
    ):
        FEATURE_BINS.labels(
            feature=feature,
            bin=str(index)
        )

        REFERENCE.labels(
            feature=feature,
            bin=str(index)
        ).set(proportion)


class MonitoredModel:
    def __init__(self, model, model_path):
        self.model = model

        self.sklearn_model = (
            mlflow.sklearn.load_model(model_path)
        )

        self.spike_index = list(
            self.sklearn_model.classes_
        ).index(1)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def predict(self, data, params=None):
        predictions = self.model.predict(
            data, params=params
        )

        try:
            probabilities = (
                self.sklearn_model.predict_proba(data)
                [:, self.spike_index]
            )

            values = np.asarray(
                predictions
            ).reshape(-1)

            if len(values) != len(probabilities):
                raise ValueError("Prediction size mismatch")

            if not np.isin(values, [0, 1]).all():
                raise ValueError("Unexpected prediction class")

            if not np.isfinite(probabilities).all():
                raise ValueError("Invalid probability")

            observations = []

            for feature, info in BASELINE["features"].items():
                bounds = info["upper_bounds"][:-1]

                for value in data[feature]:
                    value = float(value)

                    if not math.isfinite(value):
                        raise ValueError(
                            f"Invalid feature: {feature}"
                        )

                    index = bisect.bisect_left(
                        bounds, value
                    )

                    observations.append(
                        (feature, str(index))
                    )

            for prediction, probability in zip(
                values, probabilities
            ):
                label = (
                    "SPIKE"
                    if int(prediction) == 1
                    else "NO_SPIKE"
                )

                PREDICTIONS.labels(
                    prediction=label
                ).inc()

                PROBABILITY.observe(
                    float(probability)
                )

            for feature, index in observations:
                FEATURE_BINS.labels(
                    feature=feature,
                    bin=index
                ).inc()

        except Exception:
            MONITORING_ERRORS.inc()
            logger.exception(
                "Failed to collect monitoring metrics"
            )

        return predictions
