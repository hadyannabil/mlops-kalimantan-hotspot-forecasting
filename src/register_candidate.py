"""LK-08: Register a validated model to MLflow Staging."""

import hashlib
import json
import math
import os
from pathlib import Path
from validation_gate import passes_validation

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "KalimantanHotspotSpikeRF"

DATA_PATH = ROOT / "data/processed/hotspot_model_ready.csv"
SUMMARY_PATH = ROOT / "outputs/lk08_training_summary.json"
RESULT_PATH = ROOT / "outputs/lk08_validation_result.json"


def main():
    if not SUMMARY_PATH.is_file() or not RESULT_PATH.is_file():
        raise FileNotFoundError(
            "Training summary atau hasil validasi belum tersedia."
        )

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    if result.get("passed") is not True:
        raise ValueError("Model tidak lolos validasi.")

    # Prevent mismatched training and validation artifacts.
    for key in ("run_id", "model_id", "data_sha256", "git_commit"):
        if not summary.get(key) or summary[key] != result.get(key):
            raise ValueError(f"Metadata tidak sesuai: {key}")

    for key, result_key in (
        ("validation_ap", "validation_ap"),
        ("baseline_ap", "baseline_ap"),
    ):
        if not math.isclose(
            float(summary[key]),
            float(result[result_key]),
            rel_tol=0,
            abs_tol=1e-8,
        ):
            raise ValueError(f"Metrik tidak sesuai: {key}")

    # Independently enforce the performance gate before registration.
    if not passes_validation(
        float(summary["validation_ap"]),
        float(summary["baseline_ap"]),
    ):
        raise ValueError(
            "Model tidak memenuhi threshold AP untuk Staging."
        )
    current_sha = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if current_sha != summary["data_sha256"]:
        raise ValueError("Dataset tidak sesuai dengan hasil training.")

    # Check workflow commit when executed by GitHub Actions.
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha and github_sha != summary["git_commit"]:
        raise ValueError("Git commit tidak sesuai dengan workflow.")

    local_db = ROOT / "mlflow.db"
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{local_db.as_posix()}",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=tracking_uri,
    )

    run = client.get_run(summary["run_id"])
    model = client.get_logged_model(summary["model_id"])

    if run.info.status != "FINISHED":
        raise ValueError("Training run belum selesai.")

    if model.source_run_id != summary["run_id"]:
        raise ValueError("Model tidak berasal dari run terpilih.")

    if run.data.tags.get("selected_by_validation_ap") != "true":
        raise ValueError("Run belum ditandai sebagai model terpilih.")

    if run.data.tags.get("model_data_sha256") != current_sha:
        raise ValueError("Dataset pada MLflow tidak sesuai.")

    if run.data.tags.get("git_commit") != summary["git_commit"]:
        raise ValueError("Git commit pada MLflow tidak sesuai.")

    for metric, key in (
        ("val_average_precision", "validation_ap"),
        ("baseline_val_average_precision", "baseline_ap"),
    ):
        recorded = run.data.metrics.get(metric)
        if recorded is None or not math.isclose(
            float(recorded),
            float(summary[key]),
            rel_tol=0,
            abs_tol=1e-8,
        ):
            raise ValueError(f"Metrik MLflow tidak sesuai: {metric}")

    # Register the exact logged model selected by validation.
    registered = mlflow.register_model(
        model_uri=f"models:/{summary['model_id']}",
        name=MODEL_NAME,
    )

    # Stage is explicitly required by the LK-08 assignment.
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=registered.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    verified = client.get_model_version(
        MODEL_NAME, registered.version
    )

    if verified.current_stage != "Staging":
        raise RuntimeError("Transisi ke Staging gagal.")

    print("=== LK-08 REGISTRATION SUCCESS ===")
    print("Model:", MODEL_NAME)
    print("Version:", verified.version)
    print("Stage:", verified.current_stage)
    print("Run ID:", summary["run_id"])
    print("Model ID:", summary["model_id"])


if __name__ == "__main__":
    main()
