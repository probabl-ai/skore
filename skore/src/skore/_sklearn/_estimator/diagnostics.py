from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning

from skore._config import configuration
from skore._sklearn._diagnostics.base import DiagnosticResult

if TYPE_CHECKING:
    from skore._sklearn._estimator.report import EstimatorReport


_TIMING_METRICS = {"fit time (s)", "predict time (s)"}


class MetricKey(NamedTuple):
    """Unique identifier for a metric row in the summarize output."""

    metric: str
    label: str
    average: str
    output: str


@dataclass(frozen=True, slots=True)
class MetricPair:
    """A metric's train and test scores with its favorability direction."""

    favorability: str
    train: float
    test: float


def _metric_key(value: object) -> str:
    """Normalize a metric index value to a hashable string."""
    return "" if pd.isna(value) else str(value)


def _to_float(value: object) -> float | None:
    """Safely coerce `value` to float, returning ``None`` on failure."""
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _adaptive_threshold(
    *, floor: float, fraction: float, references: tuple[float, ...]
) -> float:
    """Compute a scale-aware threshold.

    Returns ``max(floor, fraction * max(|ref|..., 1.0))``.  The floor
    prevents the threshold from vanishing on near-zero scores; scaling by
    the reference magnitude keeps it meaningful for large-valued metrics.
    """
    return max(floor, fraction * max(*map(abs, references), 1.0))


def _is_significant_gap(metric: MetricPair) -> bool:
    """Check whether the train-favored gap indicates potential overfitting.

    The gap threshold is 10% of the reference score (floor 3%).
    """
    if metric.favorability == "(↗︎)":
        return metric.train - metric.test >= _adaptive_threshold(
            floor=0.03, fraction=0.10, references=(metric.train,)
        )
    if metric.favorability == "(↘︎)":
        return metric.test - metric.train >= _adaptive_threshold(
            floor=0.03, fraction=0.10, references=(metric.test,)
        )
    return False


def _is_on_par(metric: MetricPair) -> bool:
    """Check whether train and test scores are within 5% of each other (floor 3%)."""
    return abs(metric.train - metric.test) <= _adaptive_threshold(
        floor=0.03, fraction=0.05, references=(metric.train, metric.test)
    )


def _is_significantly_better(
    *, score: float, baseline: float, favorability: str
) -> bool:
    """Check whether `score` meaningfully outperforms `baseline`.

    The threshold is 3% of the baseline magnitude (floor 1%).
    """
    threshold = _adaptive_threshold(floor=0.01, fraction=0.03, references=(baseline,))
    if favorability == "(↗︎)":
        return score - baseline > threshold
    if favorability == "(↘︎)":
        return baseline - score > threshold
    return False


def _majority_triggered(votes: list[bool]) -> tuple[bool, int, int]:
    """Apply a strict-majority rule to `votes`.

    Returns ``(triggered, n_positive, n_total)``.
    """
    triggered = sum(votes)
    total = len(votes)
    return triggered > total / 2, triggered, total


def _metric_pairs(report: EstimatorReport) -> dict[MetricKey, MetricPair]:
    """Extract paired train/test scores for every predictive metric."""
    pairs: dict[MetricKey, dict[str, object]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        data = report.metrics.summarize(data_source="both").data
    for row in data.itertuples(index=False):
        metric = str(row.metric)
        if metric.lower() in _TIMING_METRICS:
            continue
        key = MetricKey(
            metric,
            _metric_key(row.label),
            _metric_key(row.average),
            _metric_key(row.output),
        )
        if key not in pairs:
            pairs[key] = {"favorability": str(row.favorability)}
        pairs[key][str(row.data_source)] = row.score
    metric_pairs: dict[MetricKey, MetricPair] = {}
    for key, values in pairs.items():
        if "train" not in values or "test" not in values:
            continue
        train_score = _to_float(values["train"])
        test_score = _to_float(values["test"])
        if train_score is None or test_score is None:
            continue
        if not (np.isfinite(train_score) and np.isfinite(test_score)):
            continue
        metric_pairs[key] = MetricPair(
            favorability=str(values["favorability"]),
            train=train_score,
            test=test_score,
        )
    return metric_pairs


def _baseline_metric_pairs(report: EstimatorReport) -> dict[MetricKey, MetricPair]:
    """Build metric pairs for a dummy baseline fitted on the same data.

    Uses ``DummyClassifier(strategy="prior")`` for classification and
    ``DummyRegressor(strategy="mean")`` for regression.
    """
    if "classification" in report.ml_task:
        dummy_estimator = DummyClassifier(strategy="prior")
    elif "regression" in report.ml_task:
        dummy_estimator = DummyRegressor(strategy="mean")
    else:
        return {}
    from skore._sklearn._estimator.report import EstimatorReport as _ER

    with configuration(diagnose=False):
        baseline_report = _ER(
            dummy_estimator,
            fit=True,
            X_train=report.X_train,
            y_train=report.y_train,
            X_test=report.X_test,
            y_test=report.y_test,
            pos_label=report.pos_label,
            diagnose=False,
        )
    return _metric_pairs(baseline_report)


@dataclass(frozen=True, slots=True)
class DiagnosticContext:
    """Pre-computed inputs shared across all diagnostic checks."""

    report: EstimatorReport
    metric_pairs: dict[MetricKey, MetricPair]
    baseline_pairs: dict[MetricKey, MetricPair]


class DiagnosticCheck(ABC):
    """A single diagnostic check that can be registered in the check list.

    Subclasses declare their identity (``code``, ``title``, ``docs_anchor``)
    and implement ``can_run`` / ``run`` to participate in the diagnostic
    pipeline without requiring changes to the orchestrator.
    """

    code: str
    title: str
    docs_anchor: str

    @abstractmethod
    def can_run(self, report: EstimatorReport) -> bool:
        """Whether the report has the data this check needs."""

    @abstractmethod
    def run(self, context: DiagnosticContext) -> DiagnosticResult | None:
        """Run the check.  Return a result if the issue is detected, else ``None``."""


def _has_train_and_test(report: EstimatorReport) -> bool:
    """Check whether the report has both train and test X/y data."""
    return (
        report.X_train is not None
        and report.y_train is not None
        and report.X_test is not None
        and report.y_test is not None
    )


class OverfittingCheck(DiagnosticCheck):
    """Detect potential overfitting via train/test score gaps.

    A metric is flagged when the train-favored gap exceeds 10% of the
    reference score (floor 3%).  The check fires when a strict majority
    of default predictive metrics are flagged.
    """

    code = "SKD001"
    title = "Potential overfitting"
    docs_anchor = "skd001-overfitting"

    def can_run(self, report: EstimatorReport) -> bool:
        return _has_train_and_test(report)

    def run(self, context: DiagnosticContext) -> DiagnosticResult | None:
        votes = [_is_significant_gap(m) for m in context.metric_pairs.values()]
        fired, triggered, total = _majority_triggered(votes)
        if not fired:
            return None
        return DiagnosticResult(
            code=self.code,
            title=self.title,
            docs_anchor=self.docs_anchor,
            explanation=(
                "Significant train/test gaps were found for "
                f"{triggered}/{total} default predictive metrics."
            ),
        )


class UnderfittingCheck(DiagnosticCheck):
    """Detect potential underfitting by comparing to a dummy baseline.

    A metric is flagged when train and test scores are within 5% of
    each other (floor 3%) AND neither exceeds the dummy baseline by
    more than 3% of the baseline (floor 1%). The check fires when a
    strict majority of comparable metrics are flagged.
    """

    code = "SKD002"
    title = "Potential underfitting"
    docs_anchor = "skd002-underfitting"

    def can_run(self, report: EstimatorReport) -> bool:
        return _has_train_and_test(report)

    def run(self, context: DiagnosticContext) -> DiagnosticResult | None:
        shared_keys = context.metric_pairs.keys() & context.baseline_pairs.keys()
        if not shared_keys:
            return None
        votes = [
            _is_on_par(context.metric_pairs[k])
            and not _is_significantly_better(
                score=context.metric_pairs[k].train,
                baseline=context.baseline_pairs[k].train,
                favorability=context.metric_pairs[k].favorability,
            )
            and not _is_significantly_better(
                score=context.metric_pairs[k].test,
                baseline=context.baseline_pairs[k].test,
                favorability=context.metric_pairs[k].favorability,
            )
            for k in shared_keys
        ]
        fired, triggered, total = _majority_triggered(votes)
        if not fired:
            return None
        return DiagnosticResult(
            code=self.code,
            title=self.title,
            docs_anchor=self.docs_anchor,
            explanation=(
                "Train/test scores are on par and not significantly better than "
                f"the dummy baseline for {triggered}/{total} comparable metrics."
            ),
        )


_CHECKS: list[DiagnosticCheck] = [OverfittingCheck(), UnderfittingCheck()]


def run_estimator_diagnostics(
    report: EstimatorReport,
) -> tuple[list[DiagnosticResult], set[str]]:
    """Run all registered diagnostic checks against `report`.

    Returns a tuple of (detected issues, set of check codes that were evaluated).
    """
    runnable = [check for check in _CHECKS if check.can_run(report)]
    if not runnable:
        return [], set()
    context = DiagnosticContext(
        report=report,
        metric_pairs=_metric_pairs(report),
        baseline_pairs=_baseline_metric_pairs(report),
    )
    return (
        [result for check in runnable if (result := check.run(context)) is not None],
        {check.code for check in runnable},
    )
