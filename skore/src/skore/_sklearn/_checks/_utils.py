from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Literal, cast

import narwhals as nw
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.pipeline import Pipeline

from skore._sklearn._plot.metrics.metrics_summary_display import MetricsSummaryRow
from skore._sklearn.types import EstimatorLike, PositiveLabel
from skore._utils._dataframe import (
    UserDataFrame,
    UserTarget,
    _concat_vertical,
    _normalize_X_as_dataframe,
    _normalize_y_as_dataframe,
)

if TYPE_CHECKING:
    from skore._sklearn._base import _BaseReport
    from skore._sklearn._cross_validation.report import CrossValidationReport
    from skore._sklearn._estimator.report import EstimatorReport
    from skore._sklearn.types import DataSource

_TIMING_METRICS = {"Fit time (s)", "Predict time (s)"}

MetricName = str
Label = PositiveLabel | None
Average = str | None
Output = int | None
MetricKey = tuple[MetricName, Label, Average, Output]

ClassName = str
ParameterName = str
StepName = str


def _metric_key(row: MetricsSummaryRow) -> MetricKey:
    """Identity tuple for a metric row (verbose name + label/average/output)."""
    return (row["metric_verbose_name"], row["label"], row["average"], row["output"])


def collect_scores(
    report: EstimatorReport | CrossValidationReport,
    *,
    data_source: DataSource,
) -> dict[MetricKey, MetricsSummaryRow]:
    """Collect ``summarize`` rows keyed by metric identity.

    For cross-validation reports, scores are mean-aggregated across splits.
    Timing rows are filtered out by default.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        rows = report.metrics.summarize(data_source=data_source).rows

    filtered_rows = [
        row for row in rows if row["metric_verbose_name"] not in _TIMING_METRICS
    ]
    if report._report_type == "estimator":
        return {_metric_key(row): row for row in filtered_rows}

    grouped: dict[MetricKey, list[MetricsSummaryRow]] = defaultdict(list)
    # There is one row per split for each metric, so we group them before averaging
    for row in filtered_rows:
        grouped[_metric_key(row)].append(row)
    return {
        key: split_rows[0]
        | {
            "score": float(np.mean([row["score"] for row in split_rows])),
        }
        for key, split_rows in grouped.items()
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
    """Raised when a check cannot run on the given report.

    Parameters
    ----------
    message : str or None, default=None
        Optional reason shown in the checks summary explanation.

    Notes
    -----
    Check implementations raise this exception when required data, task type,
    or model capabilities are missing. The check appears under the
    "Not Applicable" section of the checks summary.

    Examples
    --------
    >>> from skore import Check
    >>> from skore._sklearn._checks._utils import CheckNotApplicable
    >>> class MyCheck(Check):
    ...     code = "TST001"
    ...     title = "My check"
    ...     report_types = ["estimator"]
    ...     docs_url = None
    ...     severity = "issue"
    ...     def check_function(self, report):
    ...         if report.X_train is None:
    ...             raise CheckNotApplicable("Train data is unavailable.")
    ...         return None
    """


def split_preprocessor_estimator(estimator):
    """Return ``(preprocessor, predictor)`` from a possibly wrapped estimator.

    Splits sklearn :class:`~sklearn.pipeline.Pipeline` into its preprocessing
    steps and final predictor.
    """
    if isinstance(estimator, Pipeline):
        if len(estimator.steps) > 1:
            return estimator[:-1], estimator[-1]
        else:
            return None, estimator[0]
    return None, estimator


def cast_report(report: _BaseReport) -> EstimatorReport | CrossValidationReport:
    if report._report_type == "estimator":
        return cast("EstimatorReport", report)
    return cast("CrossValidationReport", report)


def get_report_y(
    report: EstimatorReport | CrossValidationReport,
    *,
    data_source: Literal["train", "test", "both"] = "both",
) -> UserTarget:
    """Return the target as a 1d Series or multi-output DataFrame.

    For cross-validation reports, returns the full dataset target and
    ``data_source`` is ignored.
    """
    try:
        if report._report_type == "cross-validation":
            y = nw.from_native(_normalize_y_as_dataframe(report.y))
        else:
            if data_source == "both":
                if report.y_train is None:
                    raise CheckNotApplicable("Target train data is unavailable.")
                y = nw.concat(
                    [
                        nw.from_native(_normalize_y_as_dataframe(report.y_train)),
                        nw.from_native(_normalize_y_as_dataframe(report.y_test)),
                    ],
                    how="vertical",
                )
            elif data_source == "train":
                if report.y_train is None:
                    raise CheckNotApplicable("Target train data is unavailable.")
                y = nw.from_native(_normalize_y_as_dataframe(report.y_train))
            else:
                y = nw.from_native(_normalize_y_as_dataframe(report.y_test))
        if y.shape[1] == 1:
            return y.get_column(y.columns[0]).to_native()
        return y.to_native()
    except NotImplementedError as err:
        raise CheckNotApplicable("Target data is sparse.") from err


def get_fitted_estimator(
    report: EstimatorReport | CrossValidationReport,
) -> EstimatorLike:
    if report._report_type == "cross-validation":
        return report.reports_[0].estimator_
    return report.estimator_


def get_fit_time(report: EstimatorReport | CrossValidationReport) -> float:
    if report._report_type == "cross-validation":
        return float(
            report.metrics.timings(aggregate="mean").loc["Fit time (s)", "mean"]
        )
    if report._fit_time is None:
        raise CheckNotApplicable("Fit time is unavailable.")
    return report._fit_time


def get_preprocessed_X(
    report: EstimatorReport | CrossValidationReport,
    *,
    data_source: Literal["train", "test", "both"] = "both",
) -> UserDataFrame:
    """Return the feature matrix seen by the predictor.

    Features are retrieved in the same format as at fit time, passed through
    the fitted preprocessor when present, then normalized for analysis.

    For cross-validation reports, returns features from the full dataset and
    ``data_source`` is ignored. The preprocessor is taken from the first fold's
    fitted estimator.

    Raises `CheckNotApplicable` when no data is available or when
    the preprocessor produces an unsupported type (e.g. sparse matrices).
    """
    if report._report_type == "cross-validation":
        data = report.X
    else:
        if data_source == "both":
            if report.X_train is None:
                raise CheckNotApplicable("Train data is unavailable.")
            data = _concat_vertical(report.X_train, report.X_test)
        elif data_source == "train":
            if report.X_train is None:
                raise CheckNotApplicable("Train data is unavailable.")
            data = report.X_train
        else:
            data = report.X_test

    preprocessor, _ = split_preprocessor_estimator(get_fitted_estimator(report))
    if preprocessor is not None and len(preprocessor.steps) > 0:
        data = preprocessor.transform(data)

    try:
        return _normalize_X_as_dataframe(data)
    except NotImplementedError as err:
        raise CheckNotApplicable("Feature data is sparse.") from err
