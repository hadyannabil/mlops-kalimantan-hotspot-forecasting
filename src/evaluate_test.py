"""LK-06: evaluate the selected MLflow 3 model on the final temporal test holdout.

Run from repository root after copying to src/evaluate_test.py:
    MLFLOW_ALLOW_PICKLE_DESERIALIZATION=true python src/evaluate_test.py

Only use this switch for this project's own trusted model artifacts.
No training or parameter tuning is performed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/hotspot_model_ready.csv"
METADATA = ROOT / "data/processed/model_data_metadata.json"
DATABASE = ROOT / "mlflow.db"
EXPERIMENT = "lk06-kalimantan-hotspot-random-forest"
SELECTED_RUN_ID = "0c95822681c640168f647d6f3a8e623d"


def main() -> None:
    if not DATA.is_file() or not METADATA.is_file() or not DATABASE.is_file():
        raise FileNotFoundError("Dataset, metadata, or mlflow.db not found. Run from the same Codespace.")

    mlflow.set_tracking_uri(f"sqlite:///{DATABASE.as_posix()}")
    client = MlflowClient()
    run = client.get_run(SELECTED_RUN_ID)
    if run.info.status != "FINISHED":
        raise RuntimeError("Selected run is not FINISHED.")
    if run.data.tags.get("mlflow.runName") != "rf_01_conservative":
        raise RuntimeError("Unexpected run name: confirm the selected run before testing.")

    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if run.data.tags.get("model_data_sha256") != digest:
        raise RuntimeError("Dataset checksum differs from that used during training. Stop and investigate.")
    if run.data.tags.get("final_test_evaluated") == "true":
        raise RuntimeError("Final test already completed for this run. Do not evaluate the holdout twice.")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError("Experiment not found.")

    models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id='{SELECTED_RUN_ID}'",
        output_format="list",
    )
    if len(models) != 1:
        raise RuntimeError(f"Expected exactly one logged model, found {len(models)}.")
    model_id = models[0].model_id
    model_uri = f"models:/{model_id}"
    print("Loading the model produced by the selected validation run...")
    print("Model ID:", model_id)

    # A trusted, own-project Skops model may require
    # MLFLOW_ALLOW_PICKLE_DESERIALIZATION=true in MLflow 3.
    model = mlflow.sklearn.load_model(model_uri)

    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    columns = [metadata.get("categorical_region_column", "shapeID"), *metadata["feature_columns"]]
    frame = pd.read_csv(DATA, dtype={columns[0]: "string"})
    test = frame.loc[frame["split"].eq("test")].copy()
    if len(test) != 3248:
        raise RuntimeError(f"Unexpected test size {len(test)} (expected 3248).")
    if test[columns].isna().any().any():
        raise RuntimeError("Missing values in held-out test features.")
    y = test["spike"].astype(int)
    if set(y.unique()) != {0, 1}:
        raise RuntimeError("Test does not contain both classes.")

    # Same feature list and fixed decision threshold as the three validation runs.
    probability = model.predict_proba(test[columns])[:, 1]
    predicted = (probability >= 0.5).astype(int)
    precision_curve, recall_curve, _ = precision_recall_curve(y, probability)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    results = {
        "test_average_precision": float(average_precision_score(y, probability)),
        "test_pr_auc_trapezoid": float(auc(recall_curve, precision_curve)),
        "test_roc_auc": float(roc_auc_score(y, probability)),
        "test_precision_at_0_5": float(precision_score(y, predicted, zero_division=0)),
        "test_recall_at_0_5": float(recall_score(y, predicted, zero_division=0)),
        "test_f1_at_0_5": float(f1_score(y, predicted, zero_division=0)),
        "test_spike_prevalence": float(y.mean()),
        "test_baseline_average_precision": float(y.mean()),
        "test_tn": float(tn),
        "test_fp": float(fp),
        "test_fn": float(fn),
        "test_tp": float(tp),
    }

    # Keep the selected training run and model unchanged; attach final artifacts.
    with TemporaryDirectory(prefix="lk06_test_") as folder:
        report = Path(folder) / "test_metrics.json"
        report.write_text(
            json.dumps({"run_id": SELECTED_RUN_ID, "model_id": model_id, **results}, indent=2),
            encoding="utf-8",
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y, predicted, labels=[0, 1],
            display_labels=["NO SPIKE", "SPIKE"], ax=ax, colorbar=False,
        )
        ax.set_title("Final test confusion matrix (Aug-Sep 2026)")
        fig.tight_layout()
        figure_path = Path(folder) / "test_confusion_matrix.png"
        fig.savefig(figure_path, dpi=160)
        plt.close(fig)
        client.log_artifact(SELECTED_RUN_ID, str(report), artifact_path="final_test")
        client.log_artifact(SELECTED_RUN_ID, str(figure_path), artifact_path="final_test")

    for key, value in results.items():
        client.log_metric(SELECTED_RUN_ID, key, value)
    client.set_tag(SELECTED_RUN_ID, "final_test_model_id", model_id)
    client.set_tag(SELECTED_RUN_ID, "final_test_evaluated", "true")

    print("\n=== FINAL TEST (AUG-SEP 2026) ===")
    print("Test rows:", len(test))
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
    print("Test metrics and confusion matrix logged to:", SELECTED_RUN_ID)


if __name__ == "__main__":
    main()
