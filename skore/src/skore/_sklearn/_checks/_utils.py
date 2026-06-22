from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import narwhals as nw
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.pipeline import Pipeline
from sklearn.utils._param_validation import Interval

from skore._sklearn.types import PositiveLabel
from skore._utils._dataframe import (
    UserDataFrame,
    UserTarget,
    _normalize_X_as_dataframe,
    _normalize_y_as_dataframe,
)

if TYPE_CHECKING:
    from skore._sklearn._estimator.report import EstimatorReport
    from skore._sklearn._plot.metrics.metrics_summary_display import (
        MetricsSummaryRow,
    )
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
    report: EstimatorReport,
    *,
    data_source: DataSource,
    include_timing: bool = False,
) -> dict[MetricKey, MetricsSummaryRow]:
    """Collect ``summarize`` rows keyed by metric identity for an estimator report.

    Timing rows are filtered out by default.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
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
    """Raised when a check cannot run on the given report.

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
    ...     report_type = "estimator"
    ...     docs_url = None
    ...     severity = "issue"
    ...     def check_function(self, report):
    ...         if report.X_test is None:
    ...             raise CheckNotApplicable()
    ...         return None
    """


def get_space_bound(estimator, param_name: str, side: str) -> float | None:
    """Return the closed parameter-space boundary for side ('left' or 'right').

    Navigates nested estimators (e.g. `Pipeline`) using `__`-separated names,
    then inspects `_parameter_constraints` for an
    :class:`~sklearn.utils._param_validation.Interval` whose corresponding bound is
    finite and closed (i.e. included in the domain).
    Returns `None` when no such constraint can be found.
    """
    *step_names, leaf_param = param_name.split("__")
    owner = estimator

    # Find the estimator that owns leaf_param
    for step_name in step_names:
        nested_params = owner.get_params(deep=True)
        if step_name not in nested_params:
            return None
        owner = nested_params[step_name]
    if not hasattr(owner, "_parameter_constraints"):
        return None

    # Find the Interval constraint for the leaf_param
    closed_sides_for_bound = {"left": ("left", "both"), "right": ("right", "both")}
    for constraint in owner._parameter_constraints.get(leaf_param, []):
        if not isinstance(constraint, Interval):
            continue
        bound_value = getattr(constraint, side)
        bound_is_closed = constraint.closed in closed_sides_for_bound[side]
        if bound_value is not None and bound_is_closed:
            return float(bound_value)
    return None


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


def get_report_y(
    report: EstimatorReport,
    *,
    data_source: Literal["train", "test", "both"],
) -> UserTarget | None:
    """Return the target as a 1d Series or multi-output DataFrame."""
    try:
        if data_source == "both":
            if report.y_train is None or report.y_test is None:
                return None
            y = nw.concat(
                [
                    nw.from_native(_normalize_y_as_dataframe(report.y_train)),
                    nw.from_native(_normalize_y_as_dataframe(report.y_test)),
                ],
                how="vertical",
            )
        elif data_source == "train":
            if report.y_train is None:
                return None
            y = nw.from_native(_normalize_y_as_dataframe(report.y_train))
        else:
            if report.y_test is None:
                return None
            y = nw.from_native(_normalize_y_as_dataframe(report.y_test))
        if y.shape[1] == 1:
            return y.get_column(y.columns[0]).to_native()
        return y.to_native()
    except NotImplementedError:
        return None


def get_preprocessed_X(
    report: EstimatorReport,
    *,
    data_source: Literal["train", "test", "both"],
) -> UserDataFrame | None:
    """Return the feature matrix seen by the predictor.

    When the report's estimator is a :class:`~sklearn.pipeline.Pipeline`, the
    raw feature matrix is passed through the fitted preprocessor (all steps
    except the last) before being returned.

    Returns ``None`` when no data is available or when the preprocessor
    produces an unsupported type (e.g. sparse matrices).
    """
    try:
        if data_source == "both":
            if report.X_train is None or report.X_test is None:
                return None
            data = nw.concat(
                [
                    nw.from_native(_normalize_X_as_dataframe(report.X_train)),
                    nw.from_native(_normalize_X_as_dataframe(report.X_test)),
                ],
                how="vertical",
            ).to_native()
        elif data_source == "train":
            if report.X_train is None:
                return None
            data = _normalize_X_as_dataframe(report.X_train)
        else:
            if report.X_test is None:
                return None
            data = _normalize_X_as_dataframe(report.X_test)
    except NotImplementedError:
        return None

    preprocessor, _ = split_preprocessor_estimator(report.estimator_)
    if preprocessor is not None and len(preprocessor.steps) > 0:
        data = preprocessor.transform(data)

    try:
        return _normalize_X_as_dataframe(data)
    except NotImplementedError:
        return None
