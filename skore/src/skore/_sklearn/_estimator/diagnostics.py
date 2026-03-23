from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning

from skore._config import configuration
from skore._sklearn._diagnostics.base import DiagnosticResult

if TYPE_CHECKING:
    from skore._sklearn._estimator.report import EstimatorReport


OVERFITTING_CODE = "SKD001"
UNDERFITTING_CODE = "SKD002"
_TIMING_METRICS = {"fit time (s)", "predict time (s)"}
_MetricKey = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class _MetricPair:
    favorability: str
    train: float
    test: float


def _metric_key(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _to_float(value: object) -> float | None:
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _metric_pairs(
    report: EstimatorReport,
) -> dict[_MetricKey, _MetricPair]:
    pairs: dict[_MetricKey, dict[str, object]] = {}
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
    metric_pairs: dict[_MetricKey, _MetricPair] = {}
    for key, values in pairs.items():
        if "train" not in values or "test" not in values:
            continue
        train_score = _to_float(values["train"])
        test_score = _to_float(values["test"])
        if train_score is None or test_score is None:
            continue
        if not (np.isfinite(train_score) and np.isfinite(test_score)):
            continue
        metric_pairs[key] = _MetricPair(
            favorability=str(values["favorability"]),
            train=train_score,
            test=test_score,
        )
    return metric_pairs


def _baseline_metric_pairs(
    report: EstimatorReport,
) -> dict[_MetricKey, _MetricPair]:
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


def _check_overfitting(
    metric_pairs: dict[_MetricKey, _MetricPair],
) -> DiagnosticResult | None:
    votes = [_is_significant_gap(metric) for metric in metric_pairs.values()]
    triggered = sum(votes)
    total = len(votes)
    if triggered <= total / 2:
        return None
    return DiagnosticResult(
        code=OVERFITTING_CODE,
        title="Potential overfitting",
        kind="overfitting",
        docs_anchor="skd001-overfitting",
        explanation=(
            "Significant train/test gaps were found for "
            f"{triggered}/{total} default predictive metrics."
        ),
    )


def _check_underfitting(
    metric_pairs: dict[_MetricKey, _MetricPair],
    baseline_pairs: dict[_MetricKey, _MetricPair],
) -> DiagnosticResult | None:
    shared_keys = metric_pairs.keys() & baseline_pairs.keys()
    if not shared_keys:
        return None
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
    if triggered <= total / 2:
        return None
    return DiagnosticResult(
        code=UNDERFITTING_CODE,
        title="Potential underfitting",
        kind="underfitting",
        docs_anchor="skd002-underfitting",
        explanation=(
            "Train/test scores are on par and not significantly better than "
            "the dummy baseline for "
            f"{triggered}/{total} comparable metrics."
        ),
    )


def run_estimator_diagnostics(
    report: EstimatorReport,
) -> tuple[list[DiagnosticResult], set[str]]:
    positive_results: list[DiagnosticResult] = []
    checked_codes: set[str] = set()
    if (
        report.X_train is not None
        and report.y_train is not None
        and report.X_test is not None
        and report.y_test is not None
    ):
        metric_pairs = _metric_pairs(report)
        baseline_pairs = _baseline_metric_pairs(report)
        checks = [
            _check_overfitting(metric_pairs),
            _check_underfitting(metric_pairs, baseline_pairs),
        ]
        positive_results.extend([r for r in checks if r is not None])
        checked_codes.update([OVERFITTING_CODE, UNDERFITTING_CODE])

    return positive_results, checked_codes
