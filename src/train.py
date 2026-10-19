"""LK-06: Compare three Random Forest experiments using time-based validation.

Run from the repository root:
    python src/train.py

The test split is intentionally not evaluated in this file. Choose a model using
validation first, then evaluate the selected model once in a separate step.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/hotspot_model_ready.csv"
META_PATH = ROOT / "data/processed/model_data_metadata.json"
EXPERIMENT = "lk06-kalimantan-hotspot-random-forest"
SEED = 42

CONFIGS = [
    {"name": "rf_01_conservative", "n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5},
    {"name": "rf_02_moderate", "n_estimators": 200, "max_depth": 14, "min_samples_leaf": 3},
    {"name": "rf_03_flexible", "n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> None:
    if not DATA_PATH.is_file() or not META_PATH.is_file():
        raise FileNotFoundError("File model-ready atau metadata belum tersedia. Periksa DVC.")

    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    numeric_features = metadata["feature_columns"]
    region_feature = metadata.get("categorical_region_column", "shapeID")
    input_features = [region_feature, *numeric_features]
    df = pd.read_csv(DATA_PATH, dtype={region_feature: "string"})
    missing = set(input_features + ["split", "spike"]) - set(df.columns)
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {sorted(missing)}")

    train = df.loc[df["split"] == "training"].copy()
    val = df.loc[df["split"] == "validation"].copy()
    test_count = int((df["split"] == "test").sum())
    if train.empty or val.empty or test_count == 0:
        raise ValueError("Split training, validation, atau test tidak lengkap.")
    for name, subset in [("training", train), ("validation", val)]:
        if subset[input_features].isna().any().any():
            raise ValueError(f"Terdapat fitur kosong dalam {name}.")
        if set(subset["spike"].unique()) != {0, 1}:
            raise ValueError(f"Label {name} harus mengandung kedua kelas 0 dan 1.")

    X_train = train[input_features]
    y_train = train["spike"].astype(int)
    X_val = val[input_features]
    y_val = val["spike"].astype(int)
    val_prevalence = float(y_val.mean())

    # SQLite gives MLflow a database-backed tracking store for Model Registry.
    db_path = ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment(EXPERIMENT)

    data_sha = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    revision = git_commit()
    results = []

    print(f"Training: {len(train):,} rows; SPIKE {y_train.mean():.2%}")
    print(f"Validation: {len(val):,} rows; SPIKE {val_prevalence:.2%}")
    print(f"Test: {test_count:,} rows (held out; not evaluated)")
    print(f"Baseline (constant-score) validation AP: {val_prevalence:.4f}")

    for config in CONFIGS:
        name = config["name"]
        rf_params = {
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "min_samples_leaf": config["min_samples_leaf"],
            "class_weight": "balanced_subsample",
            "random_state": SEED,
            "n_jobs": -1,
        }
        # Preprocessor is fit only to the training subset.
        prep = ColumnTransformer(
            transformers=[
                ("region", OneHotEncoder(handle_unknown="ignore"), [region_feature]),
                ("numeric", "passthrough", numeric_features),
            ],
            remainder="drop",
        )
        model = Pipeline(
            steps=[("preprocess", prep), ("random_forest", RandomForestClassifier(**rf_params))]
        )

        print(f"\nStarting {name}...", flush=True)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(rf_params)
            mlflow.log_params({
                "threshold_spike": metadata["threshold_spike"],
                "decision_threshold": 0.5,
                "target_horizon_days": 3,
                "feature_count_numeric": len(numeric_features),
                "use_region_feature": True,
                "split_strategy": "temporal_with_3_day_purge",
            })
            mlflow.set_tags({
                "git_commit": revision,
                "model_data_sha256": data_sha,
                "target_definition": metadata["target"],
                "source": "NASA FIRMS VIIRS NOAA-20 SP/NRT",
            })

            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_seconds = time.perf_counter() - start
            prob = model.predict_proba(X_val)[:, 1]
            prediction = (prob >= 0.5).astype(int)
            precision, recall, _ = precision_recall_curve(y_val, prob)
            metrics = {
                "val_average_precision": float(average_precision_score(y_val, prob)),
                "val_pr_auc_trapezoid": float(auc(recall, precision)),
                "val_roc_auc": float(roc_auc_score(y_val, prob)),
                "val_precision_at_0_5": float(precision_score(y_val, prediction, zero_division=0)),
                "val_recall_at_0_5": float(recall_score(y_val, prediction, zero_division=0)),
                "val_f1_at_0_5": float(f1_score(y_val, prediction, zero_division=0)),
                "val_spike_prevalence": val_prevalence,
                "baseline_val_average_precision": val_prevalence,
                "train_seconds": train_seconds,
            }
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(META_PATH), artifact_path="dataset_metadata")

            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_val, prediction, labels=[0, 1],
                display_labels=["NO SPIKE", "SPIKE"], ax=ax, colorbar=False,
            )
            ax.set_title(f"Validation confusion matrix — {name}")
            fig.tight_layout()
            mlflow.log_figure(fig, "validation/confusion_matrix.png")
            plt.close(fig)

            feature_names = model.named_steps["preprocess"].get_feature_names_out()
            importances = pd.DataFrame({
                "feature": feature_names,
                "importance": model.named_steps["random_forest"].feature_importances_,
            }).sort_values("importance", ascending=False)
            mlflow.log_text(importances.to_csv(index=False), "validation/feature_importance.csv")

            # One logged sklearn pipeline includes its encoder and classifier.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                input_example=X_train.head(3),
                serialization_format="skops",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
                pip_requirements=[
                    f"mlflow=={mlflow.__version__}",
                    f"scikit-learn=={sklearn.__version__}",
                    f"pandas=={pd.__version__}",
                    f"numpy=={np.__version__}",
                ],
            )
            results.append({
                "run_id": run.info.run_id,
                "name": name,
                **metrics,
            })
            print(
                f"{name} | AP={metrics['val_average_precision']:.4f} | "
                f"PR-AUC(trapz)={metrics['val_pr_auc_trapezoid']:.4f} | "
                f"Recall={metrics['val_recall_at_0_5']:.4f} | "
                f"F1={metrics['val_f1_at_0_5']:.4f}",
                flush=True,
            )

    best = max(results, key=lambda item: item["val_average_precision"])
    client = MlflowClient()
    client.set_tag(best["run_id"], "selected_by_validation_ap", "true")
    print("\n=== BEST ON VALIDATION (NOT FINAL TEST) ===")
    print("Run:", best["name"])
    print("Run ID:", best["run_id"])
    print("Validation average precision:", round(best["val_average_precision"], 4))
    print("Test has NOT been evaluated; evaluate only the selected model later.")
    print("MLflow database:", db_path)


if __name__ == "__main__":
    main()
