from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from numpy.typing import ArrayLike
from sklearn.pipeline import Pipeline

from skore._sklearn._base import Cache, _get_cached_response_values
from skore._sklearn.types import DataSource, PositiveLabel, YPlotData


def _check_all_checks(checks: list[Callable]) -> Callable:
    def check(accessor: Any) -> bool:
        return all(check(accessor) for check in checks)

    return check


def _check_has_coef(parent_estimator) -> bool:
    """Check if the estimator or its regressor_ has a specific attribute.

    This is a generic helper function. Please use the appropriate check for your report
    type.
    """
    estimator = (
        parent_estimator.steps[-1][1]
        if isinstance(parent_estimator, Pipeline)
        else parent_estimator
    )
    if hasattr(estimator, "coef_"):
        return True
    try:  # e.g. TransformedTargetRegressor()
        if hasattr(estimator.regressor_, "coef_"):
            return True
    except AttributeError as msg:
        if "object has no attribute 'regressor_'" not in str(msg):
            raise
    raise AttributeError(
        f"Estimator '{estimator}' is not a supported estimator by the function called."
    )


def _check_has_feature_importances(parent_estimator) -> bool:
    """Check if the estimator has a feature_importances_ attribute.

    This is a generic helper function. Please use the appropriate check for your report
    type.
    """
    estimator = (
        parent_estimator.steps[-1][1]
        if isinstance(parent_estimator, Pipeline)
        else parent_estimator
    )
    if hasattr(estimator, "feature_importances_"):
        return True
    raise AttributeError(
        f"Estimator '{estimator}' is not a supported estimator by the function called."
    )


def _check_roc_auc(ml_task_and_methods: list[tuple[str, list[str]]]):
    def check(accessor: Any) -> bool:
        are_supported_cases = []
        for ml_task, methods in ml_task_and_methods:
            is_supported_ml_task = ml_task in accessor._parent._ml_task
            has_methods = any(
                hasattr(accessor._parent._estimator, method) for method in methods
            )
            are_supported_cases.append(is_supported_ml_task and has_methods)

        if not any(are_supported_cases):
            err_msg = (
                f"For the task {accessor._parent._ml_task}, the estimator does not "
                "provide the right prediction methods. The called function requires "
                "the following combinations:\n\n"
            )
            err_msg += "\n".join(
                f"- {ml_task} with {methods}"
                for ml_task, methods in ml_task_and_methods
            )

            raise AttributeError(err_msg)

        return True

    return check


def _expand_data_sources(
    data_source: DataSource | Literal["both"],
) -> tuple[DataSource, ...]:
    """Expand 'both' data source to ('train', 'test')."""
    if data_source == "both":
        return ("train", "test")
    return (data_source,)


def _get_ys_for_single_report(
    *,
    cache: Cache,
    estimator_hash: int,
    estimator: Any,
    estimator_name: str,
    X: ArrayLike | None,
    y_true: ArrayLike,
    data_source: DataSource,
    response_method: str | list[str] | tuple[str, ...],
    pos_label: PositiveLabel | None,
    split: int | None = None,
) -> tuple[YPlotData, YPlotData]:
    """Get y_true and y_pred as YPlotData for a single estimator report."""
    results = _get_cached_response_values(
        cache=cache,
        estimator_hash=estimator_hash,
        estimator=estimator,
        X=X,
        response_method=response_method,
        pos_label=pos_label,
        data_source=data_source,
    )

    for key, value, is_cached in results:
        if not is_cached:
            cache[key] = value
        if key[-1] != "predict_time":
            y_pred = value

    y_true_data = YPlotData(
        estimator_name=estimator_name,
        data_source=data_source,
        split=split,
        y=y_true,
    )
    y_pred_data = YPlotData(
        estimator_name=estimator_name,
        data_source=data_source,
        split=split,
        y=y_pred,
    )

    return y_true_data, y_pred_data


########################################################################################
# Accessor related to `EstimatorReport`
########################################################################################


def _check_supported_ml_task(supported_ml_tasks: list[str]) -> Callable:
    def check(accessor: Any) -> bool:
        supported_task = any(
            task in accessor._parent._ml_task for task in supported_ml_tasks
        )

        if not supported_task:
            raise AttributeError(
                f"The {accessor._parent._ml_task} task is not a supported task by "
                f"function called. The supported tasks are {supported_ml_tasks}."
            )

        return True

    return check


def _check_estimator_has_method(method_name: str) -> Callable:
    def check(accessor: Any) -> bool:
        parent_estimator = accessor._parent.estimator_

        if hasattr(parent_estimator, method_name):
            return True

        raise AttributeError(
            f"Estimator '{parent_estimator}' is not a supported estimator by "
            f"the function called. The estimator should have a `{method_name}` "
            "method."
        )

    return check


def _check_estimator_has_coef() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if the estimator has a `coef_` attribute."""
        return _check_has_coef(accessor._parent.estimator_)

    return check


def _check_estimator_has_feature_importances() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if the estimator has a `feature_importances_` attribute."""
        return _check_has_feature_importances(accessor._parent.estimator_)

    return check


########################################################################################
# Accessor related to `CrossValidationReport`
########################################################################################


def _check_estimator_report_has_method(
    accessor_name: str,
    method_name: str,
) -> Callable:
    def check(accessor: Any) -> bool:
        estimator_report = accessor._parent.estimator_reports_[0]

        if not hasattr(estimator_report, accessor_name):
            raise AttributeError(
                f"Estimator report '{estimator_report}' does not have the "
                f"'{accessor_name}' accessor."
            )
        accessor = getattr(estimator_report, accessor_name)

        if hasattr(accessor, method_name):
            return True
        raise AttributeError(
            f"Estimator report '{estimator_report}' is not a supported estimator "
            "by the function called. The estimator report should have a report "
            f"`{method_name}` method."
        )

    return check


def _check_cross_validation_sub_estimator_has_coef() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if the underlying estimator has a `coef_` attribute."""
        return _check_has_coef(accessor._parent.estimator_reports_[0].estimator)

    return check


def _check_cross_validation_sub_estimator_has_feature_importances() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if the underlying estimator has a `feature_importances_` attribute."""
        return _check_has_feature_importances(
            accessor._parent.estimator_reports_[0].estimator_
        )

    return check


########################################################################################
# Accessor related to `ComparisonReport`
########################################################################################


def _check_comparison_report_sub_estimators_have_coef() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if all the estimators have a `coef_` attribute."""
        parent = accessor._parent
        if parent._report_type == "comparison-cross-validation":
            parent_estimators = [
                parent_report.estimator_reports_[0].estimator_
                for parent_report in parent.reports_.values()
            ]
        elif parent._report_type == "comparison-estimator":
            parent_estimators = [
                parent_report.estimator_ for parent_report in parent.reports_.values()
            ]
        else:
            raise TypeError(f"Unexpected report type: {type(parent.reports_[0])}")
        return all(_check_has_coef(e) for e in parent_estimators)

    return check


def _check_comparison_report_sub_estimators_have_feature_importances() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if all the estimators have a `feature_importances_` attribute."""
        parent = accessor._parent
        if parent._report_type == "comparison-cross-validation":
            parent_estimators = [
                parent_report.estimator_reports_[0].estimator_
                for parent_report in parent.reports_.values()
            ]
        elif parent._report_type == "comparison-estimator":
            parent_estimators = [
                parent_report.estimator_ for parent_report in parent.reports_.values()
            ]
        else:
            raise TypeError(f"Unexpected report type: {type(parent.reports_[0])}")
        return all(_check_has_feature_importances(e) for e in parent_estimators)

    return check


########################################################################################
# Accessor related to `ComparisonReport`
########################################################################################


def _check_any_sub_report_has_metric(metric: str) -> Callable[[Any], bool]:
    """Check whether any sub-report of the ComparisonReport supports `metric`."""

    def check(accessor: Any) -> bool:
        return any(
            hasattr(report.metrics, metric)
            for report in accessor._parent.reports_.values()
        )

    return check
