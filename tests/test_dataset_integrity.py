import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/hotspot_model_ready.csv"
META_PATH = ROOT / "data/processed/model_data_metadata.json"


@pytest.fixture(scope="module")
def dataset():
    return pd.read_csv(DATA_PATH, dtype={"shapeID": "string"})


@pytest.fixture(scope="module")
def metadata():
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def test_required_columns(dataset, metadata):
    """Semua fitur dan kolom penting harus tersedia."""
    features = [
        metadata["categorical_region_column"],
        *metadata["feature_columns"],
    ]
    required = set(features + ["date", "spike", "split"])
    assert required.issubset(dataset.columns)
    assert len(features) == 13


def test_no_missing_values(dataset):
    """Dataset tidak boleh mengandung missing value."""
    assert not dataset.isna().any().any()


def test_no_duplicate_observations(dataset):
    """Satu wilayah hanya boleh memiliki satu baris per tanggal."""
    assert not dataset.duplicated(
        subset=["date", "shapeID"]
    ).any()


def test_valid_labels_and_splits(dataset):
    """Label binary dan pembagian dataset harus valid."""
    assert set(dataset["spike"].unique()) == {0, 1}
    assert set(dataset["split"].unique()) == {
        "training", "validation", "test"
    }


def test_region_coverage(dataset):
    """Setiap tanggal harus mencakup 56 wilayah Kalimantan."""
    regions_per_day = dataset.groupby("date")["shapeID"].nunique()
    assert (regions_per_day == 56).all()


def test_temporal_split_and_purge(dataset):
    """Split harus kronologis dengan purge horizon 3 hari."""
    dates = pd.to_datetime(dataset["date"])

    train_dates = dates[dataset["split"] == "training"]
    val_dates = dates[dataset["split"] == "validation"]
    test_dates = dates[dataset["split"] == "test"]

    assert not train_dates.empty
    assert not val_dates.empty
    assert not test_dates.empty

    assert (val_dates.min() - train_dates.max()).days >= 4
    assert (test_dates.min() - val_dates.max()).days >= 4
