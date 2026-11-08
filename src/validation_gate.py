"""Performance validation gate for LK-08."""

import math

MIN_VALIDATION_AP = 0.60


def passes_validation(
    validation_ap: float,
    baseline_ap: float,
    threshold: float = MIN_VALIDATION_AP,
) -> bool:
    """Check whether a model is eligible for Staging."""

    for score in (validation_ap, baseline_ap, threshold):
        if not isinstance(score, (int, float)):
            raise ValueError("Metric must be numeric.")

        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Metric must be finite and within [0, 1].")

    return (
        validation_ap >= threshold
        and validation_ap > baseline_ap
    )
