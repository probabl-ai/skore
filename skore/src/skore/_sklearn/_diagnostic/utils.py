import numpy as np

_TIMING_METRICS = {
    "Fit time (s)",
    "Predict time (s)",
    "fit time (s)",
    "predict time (s)",
    "fit_time_s",
    "predict_time_s",
}


def adaptive_threshold(
    *, floor: float, fraction: float, references: tuple[float, ...]
) -> float:
    """Compute a scale-aware threshold.

    Returns ``max(floor, fraction * abs(references))``. The floor
    prevents the threshold from vanishing on near-zero scores; scaling by
    the reference magnitude keeps it meaningful for large-valued metrics.
    """
    return max(floor, fraction * max(abs(reference) for reference in references))


def check_score_gap_to_baseline(
    score: float,
    baseline: float,
    favorability: str,
    floor: float,
    fraction: float,
) -> bool:
    """Check whether `score` is significantly better than `baseline`.

    The gap threshold is `fraction` of the reference score, floored at `floor`
    to prevent the threshold from vanishing on near-zero scores.
    """
    if favorability == "(↗︎)":
        return score - baseline >= adaptive_threshold(
            floor=floor, fraction=fraction, references=(baseline,)
        )
    if favorability == "(↘︎)":
        return baseline - score >= adaptive_threshold(
            floor=floor, fraction=fraction, references=(baseline,)
        )
    return False


def majority_vote(votes: list[bool]) -> tuple[bool, int, int]:
    """Apply a strict-majority rule to `votes`.

    Returns ``(majority, n_positive, n_total)``.
    """
    n_positive = sum(votes)
    total = len(votes)
    return n_positive > total / 2, n_positive, total


def detect_outliers_mad(scores, threshold=3.5):
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    modified_z_scores = 0.6745 * (scores - median) / mad

    outliers = np.abs(modified_z_scores) > threshold
    return outliers


class DiagnosticNotApplicable(Exception):
    """Raised when a check cannot run on the given report."""
