from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from skore._sklearn._estimator.report import EstimatorReport
    from skore._sklearn._plot.metrics.metrics_summary_display import (
        MetricsSummaryRow,
    )
    from skore._sklearn.types import DataSource

_TIMING_METRICS = {"Fit time (s)", "Predict time (s)"}

MetricKey = tuple[str, Any, Any, Any]


def _metric_key(row: MetricsSummaryRow) -> MetricKey:
    """Identity tuple for a metric row (verbose name + label/average/output)."""
    return (row["metric_verbose_name"], row["label"], row["average"], row["output"])


def collect_scores(
    report: EstimatorReport,
    *,
    data_source: DataSource,
    include_timing: bool = False,
) -> dict[MetricKey, MetricsSummaryRow]:
    """Collect ``summarize`` rows keyed by metric identity for an estimator report.

    Timing rows are filtered out by default.
    """
    rows = report.metrics.summarize(data_source=data_source).rows
    return {
        _metric_key(row): row
        for row in rows
        if include_timing or row["metric_verbose_name"] not in _TIMING_METRICS
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
    greater_is_better: bool | None,
    floor: float,
    fraction: float,
) -> bool:
    """Check whether `score` is significantly better than `baseline`.

    The gap threshold is `fraction` of the reference score, floored at `floor`
    to prevent the threshold from vanishing on near-zero scores.
    """
    if pd.isna(greater_is_better):
        return False

    if greater_is_better:
        return score - baseline >= adaptive_threshold(
            floor=floor, fraction=fraction, references=(baseline,)
        )
    return baseline - score >= adaptive_threshold(
        floor=floor, fraction=fraction, references=(baseline,)
    )


def majority_vote(votes: list[bool]) -> tuple[bool, int, int]:
    """Apply a strict-majority rule to `votes`.

    Returns ``(majority, n_positive, n_total)``.
    """
    n_positive = sum(votes)
    total = len(votes)
    return n_positive > total / 2, n_positive, total


def detect_outliers_modified_zscore(scores, threshold=3):
    """Detect outliers using the modified Z-score method.

    The constant 0.6745 is a scaling factor that makes the MAD a consistent estimator
    of the standard deviation for Gaussian data, so that the resulting
    scores are comparable to ordinary Z-scores.

    See https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    if mad == 0:
        return np.zeros_like(scores)
    modified_z_scores = 0.6745 * (scores - median) / mad

    return np.abs(modified_z_scores) > threshold


class CheckNotApplicable(Exception):
    """Raised when a check cannot run on the given report."""
