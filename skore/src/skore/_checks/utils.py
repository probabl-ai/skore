from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal, cast

import narwhals as nw
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from skrub import tabular_pipeline

from skore._checks.tunable_hyperparameters import EQUIVALENT_PARAM_GROUPS
from skore._displays.metrics.metrics_summary_display import MetricsSummaryRow
from skore._sklearn.feature_names import _get_feature_names
from skore._sklearn.types import EstimatorLike, PositiveLabel
from skore._utils.dataframe import (
    UserDataFrame,
    UserTarget,
    _concat_vertical,
    _normalize_X_as_dataframe,
    _normalize_y_as_dataframe,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport
    from skore._reports.cross_validation.report import CrossValidationReport
    from skore._reports.estimator.report import EstimatorReport
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
    return (row["verbose_name"], row["label"], row["average"], row["output"])


def _summary_to_rows(summary: pd.DataFrame) -> list[MetricsSummaryRow]:
    """Convert a display summary dataframe back to metric rows."""
    nullable_cols = {
        "label",
        "average",
        "output",
        "greater_is_better",
        "split",
    }
    rows: list[MetricsSummaryRow] = []
    for record in summary.to_dict("records"):
        row: dict[str, Any] = {}
        for key, value in record.items():
            if key in nullable_cols and pd.isna(value):
                row[key] = None
            else:
                row[key] = value
        rows.append(cast("MetricsSummaryRow", row))
    return rows


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
        rows = _summary_to_rows(
            report.metrics.summarize(data_source=data_source).summary
        )

    filtered_rows = [row for row in rows if row["verbose_name"] not in _TIMING_METRICS]
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


def check_score_better_than_baseline(
    score: float,
    baseline: float,
    greater_is_better: bool | None,
    floor: float,
    fraction: float,
) -> bool:
    """Check whether `score` is significantly better than `baseline`.

    The gap threshold is ``fraction`` of the baseline score, floored at ``floor``
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


class CheckNotApplicable(Exception):
    """Raised when a check cannot run on the given report.

    Parameters
    ----------
    message : str or None, default=None
        Optional reason shown in the checks summary explanation.

    Notes
    -----
    Check implementations raise this exception when required data, ML task,
    or model capabilities are missing. The check appears under the
    "Not Applicable" section of the checks summary.

    Examples
    --------
    >>> from skore import Check
    >>> from skore._checks.utils import CheckNotApplicable
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

    preprocessor, predictor = split_preprocessor_estimator(get_fitted_estimator(report))
    if preprocessor is not None and len(preprocessor.steps) > 0:
        data = preprocessor.transform(data)
        if not nw.dependencies.is_into_dataframe(data) and not sp.issparse(data):
            data = pd.DataFrame(
                data,
                columns=_get_feature_names(
                    predictor,
                    transformer=preprocessor,
                    n_features=np.shape(data)[1],
                ),
            )

    try:
        return _normalize_X_as_dataframe(data)
    except NotImplementedError as err:
        raise CheckNotApplicable("Feature data is sparse.") from err


def baseline_estimator_report(
    report: EstimatorReport | CrossValidationReport,
    kind: Literal["dummy", "performance", "fast"],
) -> EstimatorReport | CrossValidationReport:
    """Build a baseline report mirroring ``report``.

    For ``kind="dummy"``, returns a plain ``DummyClassifier`` / ``DummyRegressor``
    baseline. For ``kind="performance"`` and ``kind="fast"``, the estimator is
    wrapped in :func:`skrub.tabular_pipeline`.

    Raises :class:`CheckNotApplicable` for unsupported ml tasks.
    """
    supported_tasks = [
        "binary-classification",
        "multiclass-classification",
        "regression",
        "multioutput-regression",
    ]
    if report.ml_task not in supported_tasks:
        raise CheckNotApplicable(
            f"Expected ML task to be one of {supported_tasks}; got {report.ml_task}."
        )
    if kind == "dummy":
        estimator = (
            DummyClassifier(strategy="prior")
            if "classification" in report.ml_task
            else DummyRegressor(strategy="mean")
        )
    elif kind == "performance":
        if "classification" in report.ml_task:
            base_estimator = HistGradientBoostingClassifier()
        elif report.ml_task == "multioutput-regression":
            base_estimator = MultiOutputRegressor(HistGradientBoostingRegressor())
        else:
            base_estimator = HistGradientBoostingRegressor()
        estimator = tabular_pipeline(base_estimator)
    else:  # kind == "fast"
        estimator = tabular_pipeline(
            LogisticRegression(max_iter=1000)
            if "classification" in report.ml_task
            else RidgeCV()
        )

    if report._report_type == "cross-validation":
        from skore._reports.cross_validation.report import CrossValidationReport

        try:
            baseline = CrossValidationReport(
                estimator,
                X=report.X,
                y=report.y,
                splitter=report.splitter,
                pos_label=report.pos_label,
                n_jobs=report.n_jobs,
            )
        except Exception as exc:
            raise CheckNotApplicable("Failed to create baseline report.") from exc
        registry = report.reports_[0]._metric_registry.copy()
        for baseline_split in baseline.reports_:
            baseline_split._metric_registry = registry
        return baseline

    if report.X_train is None:
        raise CheckNotApplicable("Train data is unavailable.")
    try:
        X_train = _normalize_X_as_dataframe(report.X_train)
        X_test = _normalize_X_as_dataframe(report.X_test)
    except NotImplementedError:
        raise CheckNotApplicable("Data is sparse.") from None

    y_train = get_report_y(report, data_source="train")
    y_test = get_report_y(report, data_source="test")
    from skore._reports.estimator.report import EstimatorReport

    try:
        baseline_report = EstimatorReport(
            estimator,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            pos_label=report.pos_label,
        )
    except Exception as exc:
        raise CheckNotApplicable("Failed to create baseline report.") from exc
    baseline_report._metric_registry = report._metric_registry
    return baseline_report


def collapse_equivalents(
    recommended: set[ParameterName], searched: set[ParameterName]
) -> set[ParameterName]:
    """Return ``recommended - searched``, collapsing equivalence groups.

    Some parameters serve the same purpose (e.g. ``max_depth``, ``min_samples_leaf``,
    ``min_samples_split``, ``max_leaf_nodes`` all limit tree depth). If any group
    member is already searched, drop the others; otherwise keep only the first
    missing member of the group.
    """
    missing = recommended - searched
    for group in EQUIVALENT_PARAM_GROUPS:
        group_set = set(group)
        if searched & group_set:
            missing -= group_set
        else:
            in_group = [param for param in group if param in missing]
            if len(in_group) > 1:
                missing -= group_set
                missing.add(in_group[0])
    return missing
