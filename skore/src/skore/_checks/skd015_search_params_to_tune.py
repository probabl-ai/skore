from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from skore._checks.base import Check
from skore._checks.tunable_hyperparameters import (
    HYPERPARAMETERS_TO_TUNE,
    get_tunable_hyperparameters,
)
from skore._checks.utils import (
    CheckNotApplicable,
    ClassName,
    ParameterName,
    StepName,
    cast_report,
    collapse_equivalents,
    get_fitted_estimator,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckSearchParamsToTune(Check):
    """Check for hyperparameters worth tuning in a search (SKD015).

    For :class:`~sklearn.model_selection.BaseSearchCV` estimators, compares the
    parameters being searched against a set of important hyperparameters and suggests
    any that are missing.

    When the search wraps a :class:`~sklearn.pipeline.Pipeline`, each step
    whose class appears in the recommendation table is checked independently,
    regardless of whether the search currently tunes any of its parameters.
    """

    code = "SKD015"
    title = "Hyperparameters worth tuning"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd015-hyperparameters-worth-tuning"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        estimator = get_fitted_estimator(report)
        if not isinstance(estimator, BaseSearchCV):
            raise CheckNotApplicable(
                "Estimator is not a BaseSearchCV instance. "
                f"Got {type(estimator).__name__}."
            )

        searched_keys = {
            key for params in estimator.cv_results_["params"] for key in params
        }
        inner_estimator = estimator.estimator
        if isinstance(inner_estimator, Pipeline):
            searched_params_by_step: dict[StepName, set[ParameterName]] = defaultdict(
                set
            )
            searched_values_by_step: dict[StepName, dict[ParameterName, set[Any]]] = (
                defaultdict(lambda: defaultdict(set))
            )
            for params in estimator.cv_results_["params"]:
                for key, val in params.items():
                    if "__" in key:
                        step_name, suffix = key.split("__", 1)
                        searched_params_by_step[step_name].add(suffix)
                        searched_values_by_step[step_name][suffix].add(val)
            searched_by_estimator: list[
                tuple[ClassName, Any, set[ParameterName], dict[ParameterName, set[Any]]]
            ] = [
                (
                    type(step).__name__,
                    step,
                    searched_params_by_step.get(name, set()),
                    searched_values_by_step.get(name, {}),
                )
                for name, step in inner_estimator.steps
                if type(step).__name__ in HYPERPARAMETERS_TO_TUNE
            ]
            if not searched_by_estimator:
                raise CheckNotApplicable(
                    "No parameter to recommend for any of the steps."
                )
        else:
            class_name = type(inner_estimator).__name__
            if class_name not in HYPERPARAMETERS_TO_TUNE:
                raise CheckNotApplicable("No parameter to recommend for the estimator.")
            searched_values: dict[ParameterName, set[Any]] = defaultdict(set)
            for params in estimator.cv_results_["params"]:
                for key, val in params.items():
                    searched_values[key].add(val)
            searched_by_estimator = [
                (class_name, inner_estimator, searched_keys, searched_values)
            ]

        messages: list[str] = []
        for class_name, step, searched, searched_values in searched_by_estimator:
            recommended = get_tunable_hyperparameters(step, searched_values)
            missing = collapse_equivalents(recommended, searched)
            if missing:
                messages.append(f"{sorted(missing)} for {class_name}")
        if not messages:
            return None
        messages.sort()
        return (
            "These hyperparameters are not in the grid and may be worth tuning: "
            f"{'; '.join(messages)}."
        )
