import math

import pytest

from src.validation_gate import passes_validation


@pytest.mark.parametrize(
    "ap, baseline, expected",
    [
        (0.6188, 0.1798, True),
        (0.5763, 0.1798, False),
        (0.6000, 0.1798, True),
        (0.6100, 0.7000, False),
        (0.5000, 0.5000, False),
    ],
)
def test_performance_gate(ap, baseline, expected):
    assert passes_validation(ap, baseline) is expected


@pytest.mark.parametrize(
    "invalid_ap",
    [float("nan"), math.inf, -0.1, 1.1],
)
def test_invalid_metrics(invalid_ap):
    with pytest.raises(ValueError):
        passes_validation(invalid_ap, 0.1798)
