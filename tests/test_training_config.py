import pytest

from src.train import CONFIGS, SEED


def test_random_state():
    """Memastikan random seed tetap konsisten."""
    assert SEED == 42


def test_unique_model_names():
    """Memastikan setiap konfigurasi memiliki nama unik."""
    names = [config["name"] for config in CONFIGS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("config", CONFIGS)
def test_valid_hyperparameters(config):
    """Memastikan hyperparameter Random Forest valid."""
    assert config["n_estimators"] > 0
    assert config["min_samples_leaf"] >= 1

    if config["max_depth"] is not None:
        assert config["max_depth"] >= 1
