import os
from pathlib import Path

import mlflow
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(ROOT / "docker_assets/model"))
)

app = Flask(__name__)

# Muat model sekali saat aplikasi dijalankan
model = mlflow.pyfunc.load_model(str(MODEL_PATH))

MODEL_NAME = "KalimantanHotspotSpikeRF"
MODEL_VERSION = 1

# Kontrak input sesuai MLflow Model Signature
SCHEMA = {
    "shapeID": "str",
    "hotspot_count": "int",
    "delta": "float",
    "lag_1": "float",
    "lag_2": "float",
    "lag_3": "float",
    "lag_7": "float",
    "rolling_mean_3": "float",
    "rolling_mean_7": "float",
    "rolling_max_7": "float",
    "rolling_std_7": "float",
    "month": "int",
    "day_of_year": "int",
}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "version": MODEL_VERSION
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Body harus JSON object"}), 400

    records = payload.get("instances")

    if not isinstance(records, list) or not 1 <= len(records) <= 100:
        return jsonify({
            "error": "instances harus berisi 1 sampai 100 observasi"
        }), 400

    try:
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Input {index} bukan object")

            missing = set(SCHEMA) - set(record)
            extra = set(record) - set(SCHEMA)

            if missing or extra:
                raise ValueError(
                    f"Input {index}: missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )

            for col, kind in SCHEMA.items():
                value = record[col]

                if kind == "str":
                    valid = isinstance(value, str)
                elif kind == "int":
                    valid = type(value) is int
                else:
                    valid = type(value) in (int, float)

                if not valid:
                    raise ValueError(
                        f"Input {index}: tipe fitur {col} tidak sesuai"
                    )

        df = pd.DataFrame(records, columns=list(SCHEMA))

        for col, kind in SCHEMA.items():
            if kind == "str":
                df[col] = df[col].astype(str)
            elif kind == "int":
                df[col] = df[col].astype("int64")
            else:
                df[col] = df[col].astype("float64")

        if not pd.api.types.is_numeric_dtype(
            df.drop(columns=["shapeID"]).dtypes.iloc[0]
        ):
            raise ValueError("Fitur numerik tidak valid")

        import numpy as np
        if not np.isfinite(
            df.drop(columns=["shapeID"]).to_numpy(dtype=float)
        ).all():
            raise ValueError("Input numerik harus bernilai finite")

        predictions = model.predict(df)

        results = [
            {
                "prediction": int(value),
                "status": "SPIKE" if int(value) == 1 else "NO SPIKE"
            }
            for value in predictions
        ]

        return jsonify({
            "model": MODEL_NAME,
            "version": MODEL_VERSION,
            "predictions": results
        })

    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )
