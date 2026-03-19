from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning

from skore._sklearn._diagnostics.base import DiagnosticResult

if TYPE_CHECKING:
    from skore._sklearn._estimator.report import EstimatorReport


OVERFITTING_CODE = "SKD001"
UNDERFITTING_CODE = "SKD002"
_TIMING_METRICS = {"fit time (s)", "predict time (s)"}


@dataclass(frozen=True, slots=True)
class _MetricPair:
    favorability: str
    train: float
    test: float


def _metric_key(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _metric_pairs(
    report: EstimatorReport,
) -> dict[tuple[str, str, str, str], _MetricPair]:
    pairs: dict[tuple[str, str, str, str], dict[str, object]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        data = report.metrics.summarize(data_source="both").data
    for row in data.itertuples(index=False):
        metric = str(row.metric)
        if metric.lower() in _TIMING_METRICS:
            continue
        key = (
            metric,
            _metric_key(row.label),
            _metric_key(row.average),
            _metric_key(row.output),
        )
        if key not in pairs:
            pairs[key] = {"favorability": str(row.favorability)}
        pairs[key][str(row.data_source)] = row.score
    metric_pairs: dict[tuple[str, str, str, str], _MetricPair] = {}
    for key, values in pairs.items():
        if "train" not in values or "test" not in values:
            continue
        try:
            train_score = float(values["train"])
            test_score = float(values["test"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(train_score) and np.isfinite(test_score)):
            continue
        metric_pairs[key] = _MetricPair(
            favorability=str(values["favorability"]),
            train=train_score,
            test=test_score,
        )
    return metric_pairs


def _gap_threshold(reference: float) -> float:
    return max(0.03, 0.10 * max(abs(reference), 1.0))


def _is_significant_gap(metric: _MetricPair) -> bool:
    if metric.favorability == "(↗︎)":
        return metric.train - metric.test >= _gap_threshold(metric.train)
    if metric.favorability == "(↘︎)":
        return metric.test - metric.train >= _gap_threshold(metric.test)
    return False


def _parity_threshold(metric: _MetricPair) -> float:
    return max(0.03, 0.05 * max(abs(metric.train), abs(metric.test), 1.0))


def _is_on_par(metric: _MetricPair) -> bool:
    return abs(metric.train - metric.test) <= _parity_threshold(metric)


def _is_significantly_better(
    *,
    score: float,
    baseline: float,
    favorability: str,
) -> bool:
    threshold = max(0.01, 0.03 * max(abs(baseline), 1.0))
    if favorability == "(↗︎)":
        return score - baseline > threshold
    if favorability == "(↘︎)":
        return baseline - score > threshold
    return False


def _create_dummy_report(report: EstimatorReport) -> EstimatorReport | None:
    if (
        report.X_train is None
        or report.y_train is None
        or report.X_test is None
        or report.y_test is None
    ):
        return None
    if "classification" in report.ml_task:
        dummy_estimator = DummyClassifier(strategy="prior")
    elif "regression" in report.ml_task:
        dummy_estimator = DummyRegressor(strategy="mean")
    else:
        return None
    from skore._sklearn._estimator.report import EstimatorReport

    kwargs: dict[str, object] = {
        "fit": True,
        "X_train": report.X_train,
        "y_train": report.y_train,
        "X_test": report.X_test,
        "y_test": report.y_test,
        "pos_label": report.pos_label,
    }
    if "diagnose" in inspect.signature(EstimatorReport).parameters:
        kwargs["diagnose"] = None
    try:
        return EstimatorReport(dummy_estimator, **kwargs)
    except Exception:
        return None


def _overfitting_result(
    *,
    metric_pairs: dict[tuple[str, str, str, str], _MetricPair],
    evaluated: bool,
    explanation: str,
) -> DiagnosticResult:
    if not evaluated:
        return DiagnosticResult(
            code=OVERFITTING_CODE,
            title="Potential overfitting",
            kind="overfitting",
            docs_anchor="skd001-overfitting",
            explanation=explanation,
            is_issue=False,
            evaluated=False,
        )
    votes = [_is_significant_gap(metric) for metric in metric_pairs.values()]
    triggered = sum(votes)
    total = len(votes)
    is_issue = triggered > total / 2
    if triggered == 0:
        explanation = f"No significant train/test gap was found across {total} default predictive metrics."
    elif is_issue:
        explanation = f"Significant train/test gaps were found for {triggered}/{total} default predictive metrics."
    else:
        explanation = (
            f"Significant train/test gaps were found for {triggered}/{total} default predictive metrics, "
            "which is below the majority threshold."
        )
    return DiagnosticResult(
        code=OVERFITTING_CODE,
        title="Potential overfitting",
        kind="overfitting",
        docs_anchor="skd001-overfitting",
        explanation=explanation,
        is_issue=is_issue,
        evaluated=True,
    )


def _underfitting_result(
    *,
    metric_pairs: dict[tuple[str, str, str, str], _MetricPair],
    baseline_pairs: dict[tuple[str, str, str, str], _MetricPair],
    evaluated: bool,
    explanation: str,
) -> DiagnosticResult:
    if not evaluated:
        return DiagnosticResult(
            code=UNDERFITTING_CODE,
            title="Potential underfitting",
            kind="underfitting",
            docs_anchor="skd002-underfitting",
            explanation=explanation,
            is_issue=False,
            evaluated=False,
        )
    shared_keys = metric_pairs.keys() & baseline_pairs.keys()
    if not shared_keys:
        return DiagnosticResult(
            code=UNDERFITTING_CODE,
            title="Potential underfitting",
            kind="underfitting",
            docs_anchor="skd002-underfitting",
            explanation="No shared predictive metrics were available to compare against the dummy baseline.",
            is_issue=False,
            evaluated=False,
        )
    votes = []
    for key in shared_keys:
        metric = metric_pairs[key]
        baseline = baseline_pairs[key]
        votes.append(
            _is_on_par(metric)
            and not _is_significantly_better(
                score=metric.train,
                baseline=baseline.train,
                favorability=metric.favorability,
            )
            and not _is_significantly_better(
                score=metric.test,
                baseline=baseline.test,
                favorability=metric.favorability,
            )
        )
    triggered = sum(votes)
    total = len(votes)
    is_issue = triggered > total / 2
    if triggered == 0:
        explanation = (
            f"Train and test scores are meaningfully better than the dummy baseline for all {total} "
            "comparable metrics."
        )
    elif is_issue:
        explanation = (
            f"Train/test scores are on par and not significantly better than the dummy baseline for "
            f"{triggered}/{total} comparable metrics."
        )
    else:
        explanation = (
            f"Train/test scores are on par and not significantly better than the dummy baseline for "
            f"{triggered}/{total} comparable metrics, which is below the majority threshold."
        )
    return DiagnosticResult(
        code=UNDERFITTING_CODE,
        title="Potential underfitting",
        kind="underfitting",
        docs_anchor="skd002-underfitting",
        explanation=explanation,
        is_issue=is_issue,
        evaluated=True,
    )


def run_estimator_diagnostics(
    report: EstimatorReport,
    *,
    expensive: bool = False,
) -> list[DiagnosticResult]:
    if (
        report.X_train is None
        or report.y_train is None
        or report.X_test is None
        or report.y_test is None
    ):
        explanation = "Train and test data are both required to run this diagnostic."
        return [
            _overfitting_result(
                metric_pairs={}, evaluated=False, explanation=explanation
            ),
            _underfitting_result(
                metric_pairs={},
                baseline_pairs={},
                evaluated=False,
                explanation=explanation,
            ),
        ]
    try:
        metric_pairs = _metric_pairs(report)
    except Exception as error:
        explanation = f"Failed to compute report metrics for diagnostics: {error}."
        return [
            _overfitting_result(
                metric_pairs={}, evaluated=False, explanation=explanation
            ),
            _underfitting_result(
                metric_pairs={},
                baseline_pairs={},
                evaluated=False,
                explanation=explanation,
            ),
        ]
    if not metric_pairs:
        explanation = (
            "No predictive metrics were available to evaluate this diagnostic."
        )
        return [
            _overfitting_result(
                metric_pairs={}, evaluated=False, explanation=explanation
            ),
            _underfitting_result(
                metric_pairs={},
                baseline_pairs={},
                evaluated=False,
                explanation=explanation,
            ),
        ]
    baseline_report = _create_dummy_report(report)
    baseline_pairs: dict[tuple[str, str, str, str], _MetricPair] = {}
    if baseline_report is not None:
        try:
            baseline_pairs = _metric_pairs(baseline_report)
        except Exception:
            baseline_pairs = {}
    underfitting_evaluated = bool(baseline_pairs)
    underfitting_explanation = (
        "A dummy baseline could not be computed for this report."
        if not underfitting_evaluated
        else ""
    )
    diagnostics = [
        _overfitting_result(metric_pairs=metric_pairs, evaluated=True, explanation=""),
        _underfitting_result(
            metric_pairs=metric_pairs,
            baseline_pairs=baseline_pairs,
            evaluated=underfitting_evaluated,
            explanation=underfitting_explanation,
        ),
    ]
    if expensive:
        return diagnostics
    return diagnostics
