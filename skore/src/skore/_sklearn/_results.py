from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import pandas as pd

from skore._sklearn._plot.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)

if TYPE_CHECKING:
    from skore._sklearn._cross_validation.report import CrossValidationReport
    from skore._sklearn._estimator.report import EstimatorReport
    from skore._sklearn.metrics import Metric


_DISPLAY_METHODS = (
    ("roc", ("train", "test", "both")),
    ("precision_recall", ("train", "test", "both")),
    ("prediction_error", ("train", "test", "both")),
    ("confusion_matrix", ("train", "test")),
)

_INSPECTION_METHODS = ("coefficients", "impurity_decrease")
_ResultDict = dict[str, Any]


def _infer_metric_label(*, report: EstimatorReport, metric: Metric) -> Any:
    """Infer the effective label used by scalar classification metrics."""
    if report.ml_task != "binary-classification" or report.pos_label is None:
        return None

    if metric.kwargs.get("average") == "binary":
        return report.pos_label

    if metric.name in {"precision", "recall"} and "average" not in metric.kwargs:
        return report.pos_label

    return None


# FIXME: once skore-hub-project
def _restore_sanitized_key(value: Any) -> Any:
    """Best-effort restoration of a sanitized cache key value."""
    if not isinstance(value, tuple):
        return value
    if len(value) == 2 and value[0] == "mapping":
        return {key: _restore_sanitized_key(item) for key, item in value[1]}
    if len(value) == 2 and value[0] == "set":
        return [_restore_sanitized_key(item) for item in value[1]]
    if len(value) == 2 and value[0] in {"array", "callable"}:
        return {"type": value[0], "hash": value[1]}
    if len(value) == 2 and value[0] == "dtype":
        return {"type": "dtype", "value": value[1]}
    return [_restore_sanitized_key(item) for item in value]


def _result_rows_from_metric_value(
    *,
    report: EstimatorReport,
    metric: Metric,
    data_source: str,
    value: Any,
    aggregate: str | None = None,
) -> list[dict[str, Any]]:
    base_row = {
        "data_source": data_source,
        "name": metric.name,
        "verbose_name": metric.verbose_name,
        "greater_is_better": metric.greater_is_better,
        "kwargs": metric.kwargs.copy(),
        "label": None,
        "output": None,
        "value": value,
    }
    if aggregate is not None:
        base_row["aggregate"] = aggregate

    if isinstance(value, dict):
        return [
            {**base_row, "label": label, "value": score}
            for label, score in value.items()
        ]

    if isinstance(value, list):
        return [
            {**base_row, "output": output_idx, "value": score}
            for output_idx, score in enumerate(value)
        ]

    metric_label = _infer_metric_label(report=report, metric=metric)
    if metric_label is not None:
        base_row["label"] = metric_label

    return [base_row]


def _aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    templates: dict[tuple[Any, ...], dict[str, Any]] = {}

    for row in rows:
        key = (
            row["data_source"],
            row["name"],
            row["label"],
            row["output"],
        )
        templates[key] = {
            key_: value
            for key_, value in row.items()
            if key_ not in {"aggregate", "value"}
        }
        templates[key]["kwargs"] = row["kwargs"].copy()
        grouped[key].append(row["value"])

    aggregated_rows = []
    for key, values in grouped.items():
        series = pd.Series(values, dtype="float64")
        aggregated_rows.extend(
            [
                {
                    **templates[key],
                    "aggregate": aggregate,
                    "value": series.aggregate(aggregate),
                }
                for aggregate in ("mean", "std")
            ]
        )
    return aggregated_rows


def _collect_metric_results(report: EstimatorReport) -> list[dict[str, Any]]:
    report.cache_predictions()

    data_sources = []
    if report.X_train is not None and report.y_train is not None:
        data_sources.append("train")
    if report.X_test is not None and report.y_test is not None:
        data_sources.append("test")

    results: list[dict[str, Any]] = []
    for data_source in data_sources:
        for metric in report._metric_registry.values():
            value = metric(
                report=report,
                data_source=data_source,
                **metric.kwargs,
            )
            results.extend(
                _result_rows_from_metric_value(
                    report=report,
                    metric=metric,
                    data_source=data_source,
                    value=value,
                )
            )

    return results


def _collect_display_results(
    report: EstimatorReport | CrossValidationReport,
) -> list[_ResultDict]:
    available_data_sources = {"test"}
    if hasattr(report, "X_train") and report.X_train is not None:
        available_data_sources.add("train")
        available_data_sources.add("both")
    elif not hasattr(report, "X_train"):
        available_data_sources.update({"train", "both"})

    results = []
    for method_name, data_sources in _DISPLAY_METHODS:
        if not hasattr(report.metrics, method_name):
            continue
        method = getattr(report.metrics, method_name)
        for data_source in data_sources:
            if data_source not in available_data_sources:
                continue
            try:
                display = method(data_source=data_source)
            except (NotImplementedError, TypeError, ValueError):
                continue
            results.append(
                {
                    "data_source": data_source,
                    "name": method_name,
                    "display": display,
                }
            )

    return results


def _collect_estimator_inspection_results(
    report: EstimatorReport,
) -> list[dict[str, Any]]:
    results = []

    for method_name in _INSPECTION_METHODS:
        if not hasattr(report.inspection, method_name):
            continue
        results.append(
            {
                "data_source": None,
                "name": method_name,
                "kwargs": {},
                "display": getattr(report.inspection, method_name)(),
            }
        )

    for key, display in report._cache.items():
        if (
            len(key) != 3
            or key[1] != "permutation_importance"
            or not isinstance(display, PermutationImportanceDisplay)
        ):
            continue
        data_source, name, kwargs = key
        results.append(
            {
                "data_source": data_source,
                "name": name,
                "kwargs": _restore_sanitized_key(kwargs),
                "display": display,
            }
        )

    return sorted(
        results,
        key=lambda row: (
            row["name"],
            str(row["data_source"]),
            repr(row["kwargs"]),
        ),
    )


def _collect_cross_validation_metric_results(
    report: CrossValidationReport,
) -> list[dict[str, Any]]:
    report.cache_predictions()

    results = []
    data_sources = ("train", "test")

    for data_source in data_sources:
        for metric_name in report.estimator_reports_[0]._metric_registry:
            split_rows = []
            for estimator_report in report.estimator_reports_:
                split_metric = estimator_report._metric_registry[metric_name]
                value = split_metric(
                    report=estimator_report,
                    data_source=data_source,
                    **split_metric.kwargs,
                )
                split_rows.extend(
                    _result_rows_from_metric_value(
                        report=estimator_report,
                        metric=split_metric,
                        data_source=data_source,
                        value=value,
                    )
                )
            results.extend(_aggregate_metric_rows(split_rows))

    return results


def _collect_cross_validation_inspection_results(
    report: CrossValidationReport,
) -> list[dict[str, Any]]:
    results = []

    for method_name in _INSPECTION_METHODS:
        if not hasattr(report.inspection, method_name):
            continue
        results.append(
            {
                "data_source": None,
                "name": method_name,
                "kwargs": {},
                "display": getattr(report.inspection, method_name)(),
            }
        )

    permutation_importances: dict[
        tuple[str, Any], list[tuple[int, PermutationImportanceDisplay]]
    ] = defaultdict(list)
    for split_idx, estimator_report in enumerate(report.estimator_reports_):
        for key, display in estimator_report._cache.items():
            if (
                len(key) != 3
                or key[1] != "permutation_importance"
                or not isinstance(display, PermutationImportanceDisplay)
            ):
                continue
            permutation_importances[(key[0], key[2])].append((split_idx, display))

    for (data_source, kwargs), displays in permutation_importances.items():
        if len(displays) != len(report.estimator_reports_):
            continue
        frames = [
            display.importances.assign(split=split_idx)
            for split_idx, display in sorted(displays, key=lambda item: item[0])
        ]
        results.append(
            {
                "data_source": data_source,
                "name": "permutation_importance",
                "kwargs": _restore_sanitized_key(kwargs),
                "display": PermutationImportanceDisplay(
                    importances=pd.concat(frames, ignore_index=True),
                    report_type="cross-validation",
                ),
            }
        )

    return sorted(
        results,
        key=lambda row: (
            row["name"],
            str(row["data_source"]),
            repr(row["kwargs"]),
        ),
    )


def get_estimator_report_results(
    report: EstimatorReport,
) -> dict[str, list[_ResultDict]]:
    """Collect metrics, metric displays, and inspection displays for a report."""
    return {
        "metrics": _collect_metric_results(report),
        "displays": _collect_display_results(report),
        "inspection": _collect_estimator_inspection_results(report),
    }


def get_cross_validation_report_results(
    report: CrossValidationReport,
) -> dict[str, list[_ResultDict]]:
    """Collect metrics, metric displays, and inspection displays for a CV report."""
    return {
        "metrics": _collect_cross_validation_metric_results(report),
        "displays": _collect_display_results(report),
        "inspection": _collect_cross_validation_inspection_results(report),
    }
