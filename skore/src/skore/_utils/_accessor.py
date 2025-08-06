from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from sklearn.pipeline import Pipeline


def _check_all_checks(checks: list[Callable]) -> Callable:
    def check(accessor: Any) -> bool:
        return all(check(accessor) for check in checks)

    return check


def _check_parent_estimator_has_coef(parent_estimator) -> bool:
    """Check if estimator or its regressor_ has a specific attribute."""
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
        f"Estimator {estimator} is not a supported estimator by the function called."
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
            f"Estimator {parent_estimator} is not a supported estimator by "
            f"the function called. The estimator should have a `{method_name}` "
            "method."
        )

    return check


def _check_has_coef() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if the estimator has a `coef_` attribute."""
        return _check_parent_estimator_has_coef(accessor._parent.estimator_)

    return check


def _check_has_feature_importances() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if the estimator has a `feature_importances_` attribute."""
        parent_estimator = accessor._parent.estimator_
        estimator = (
            parent_estimator.steps[-1][1]
            if isinstance(parent_estimator, Pipeline)
            else parent_estimator
        )
        if hasattr(estimator, "feature_importances_"):
            return True
        raise AttributeError(
            f"Estimator {parent_estimator} is not a supported estimator by "
            "the function called."
        )

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
                f"Estimator report {estimator_report} does not have the "
                f"'{accessor_name}' accessor."
            )
        accessor = getattr(estimator_report, accessor_name)

        if hasattr(accessor, method_name):
            return True
        raise AttributeError(
            f"Estimator report {estimator_report} is not a supported estimator report "
            "by the function called. The estimator report should have a "
            f"`{method_name}` method."
        )

    return check


def _check_estimator_report_has_coef() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if the underlying estimator has a `coef_` attribute."""
        return _check_parent_estimator_has_coef(
            accessor._parent.estimator_reports_[0].estimator
        )

    return check


########################################################################################
# Accessor related to `ComparisonReport`
########################################################################################


def _check_report_estimators_have_coef() -> Callable:
    def check(accessor: Any) -> bool:
        """Check if all the estimators have a `coef_` attribute."""
        from skore import CrossValidationReport, EstimatorReport

        parent = accessor._parent
        parent_estimators = []
        for parent_report in parent.reports_:
            if isinstance(parent_report, CrossValidationReport):
                parent_report = cast(CrossValidationReport, parent_report)
                parent_estimators.append(parent_report.estimator_reports_[0].estimator_)
            elif isinstance(parent.reports_[0], EstimatorReport):
                parent_estimators.append(parent_report.estimator_)
            else:
                raise TypeError(f"Unexpected report type: {type(parent.reports_[0])}")
        return all(_check_parent_estimator_has_coef(e) for e in parent_estimators)

    return check
