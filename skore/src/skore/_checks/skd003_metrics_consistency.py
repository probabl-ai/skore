from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from skore._checks.base import Check
from skore._checks.utils import majority_vote

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport
    from skore._reports.cross_validation.report import CrossValidationReport

_TIMING_METRICS_FLAT = {"fit_time", "predict_time"}


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


class CheckMetricsConsistencyAcrossSplits(Check):
    """Check the consistency of metrics across splits (SKD003).

    Outlier splits are identified with a modified Z-score based on the
    Median Absolute Deviation (MAD) to be robust to extreme values.
    """

    code = "SKD003"
    title = "Inconsistent performance across splits"
    report_types = ["cross-validation"]
    docs_url = "skd003-inconsistent-performance"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect outlier performance across cross-validation splits."""
        report = cast("CrossValidationReport", report)

        report_data = report.metrics.summarize(data_source="test").frame(
            aggregate=None, flat_index=True
        )
        votes = np.array(
            [
                detect_outliers_modified_zscore(report_data.loc[idx])
                for idx in report_data.index
                if idx not in _TIMING_METRICS_FLAT
            ]
        )
        explanation = []
        for cv in range(report_data.shape[1]):
            majority, n_positive, total = majority_vote(votes[:, cv].tolist())
            if majority:
                explanation.append(f"in split #{cv} for {n_positive}/{total} metrics")
        if explanation:
            return "Performance is abnormal " + " and ".join(explanation) + "."
        return None
