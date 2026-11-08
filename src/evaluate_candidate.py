"""LK-08: Validate a trained model against MLflow metrics."""

import hashlib
import json
import math
import os
from pathlib import Path

from mlflow.tracking import MlflowClient
from validation_gate import passes_validation


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data/processed/hotspot_model_ready.csv"
SUMMARY_PATH = ROOT / "outputs/lk08_training_summary.json"
RESULT_PATH = ROOT / "outputs/lk08_validation_result.json"


def main():
    # Prevent an old successful result from being reused.
    RESULT_PATH.unlink(missing_ok=True)

    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError("Training summary belum tersedia.")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    required = {
        "run_id", "run_name", "model_id",
        "validation_ap", "baseline_ap",
        "data_sha256", "git_commit"
    }

    if not required.issubset(summary):
        raise ValueError("Training summary tidak lengkap.")

    # Verify that the dataset matches the training snapshot.
    current_sha = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    if current_sha != summary["data_sha256"]:
        raise ValueError("Dataset SHA256 tidak sesuai.")

    # Connect to the same MLflow server as training.
    local_db = ROOT / "mlflow.db"
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{local_db.as_posix()}"
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(summary["run_id"])
    # Verify that the selected model belongs to this training run.
    model_id = summary["model_id"]

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Model ID tidak valid.")

    logged_model = client.get_logged_model(model_id)

    if logged_model.source_run_id != summary["run_id"]:
        raise ValueError(
            "Model ID tidak berasal dari training run yang terpilih."
        )

    if run.info.status != "FINISHED":
        raise ValueError("MLflow run belum berhasil diselesaikan.")

    metrics = run.data.metrics
    tags = run.data.tags

    # Confirm the correct run, Git revision and dataset.
    expected_tags = {
        "mlflow.runName": summary["run_name"],
        "model_data_sha256": current_sha,
        "git_commit": summary["git_commit"],
        "selected_by_validation_ap": "true",
    }

    for key, expected in expected_tags.items():
        if tags.get(key) != expected:
            raise ValueError(f"MLflow tag mismatch: {key}")

    metric_pairs = {
        "val_average_precision": summary["validation_ap"],
        "baseline_val_average_precision": summary["baseline_ap"],
    }

    for key, expected in metric_pairs.items():
        actual = metrics.get(key)

        if actual is None or not math.isclose(
            float(actual), float(expected),
            rel_tol=0,
            abs_tol=1e-8
        ):
            raise ValueError(f"MLflow metric mismatch: {key}")

    ap = float(summary["validation_ap"])
    baseline = float(summary["baseline_ap"])

    passed = passes_validation(ap, baseline)

    print("=== LK-08 MODEL VALIDATION ===")
    print("Run ID:", summary["run_id"])
    print("Validation AP:", ap)
    print("Baseline AP:", baseline)
    print("Validation passed:", passed)

    if not passed:
        raise SystemExit("VALIDATION FAILED: registrasi ditolak.")

    result = {
        "run_id": summary["run_id"],
        "model_id": model_id,
        "validation_ap": ap,
        "baseline_ap": baseline,
        "data_sha256": current_sha,
        "git_commit": summary["git_commit"],
        "passed": True,
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("VALIDATION PASSED")
    print("Result:", RESULT_PATH)


if __name__ == "__main__":
    main()
