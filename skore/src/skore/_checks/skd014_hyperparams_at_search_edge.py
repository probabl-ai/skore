from __future__ import annotations

import numbers
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils._param_validation import Interval

from skore._checks.base import Check
from skore._checks.utils import CheckNotApplicable, cast_report, get_fitted_estimator

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


def _get_space_bound(
    estimator, *, param_name: str, side: Literal["left", "right"]
) -> float | None:
    """Fetch the closed parameter-space boundary for `side` if it exists."""
    *step_names, leaf_param = param_name.split("__")
    owner = estimator
    for step_name in step_names:
        owner = owner.get_params(deep=True)[step_name]
    if not hasattr(owner, "_parameter_constraints"):
        return None
    for constraint in owner._parameter_constraints[leaf_param]:
        if isinstance(constraint, Interval) and constraint.closed in [side, "both"]:
            return float(getattr(constraint, side))
    return None


class CheckHyperparamsAtSearchEdge(Check):
    """Check whether tuned hyperparameters sit at the search boundary (SKD014).

    For :class:`~sklearn.model_selection.BaseSearchCV` estimators, flags when any
    numeric ``best_params_`` value equals the minimum or maximum distinct value
    tried for that parameter. Non-numeric hyperparameters (``bool``, strings, ``None``,
    and similar) are skipped because extending the search range is not meaningful.
    """

    code = "SKD014"
    title = "Hyperparameters at search edge"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd014-hyperparams-at-search-edge"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        estimator = get_fitted_estimator(report)
        if not isinstance(estimator, BaseSearchCV):
            raise CheckNotApplicable(
                "Estimator is not a BaseSearchCV instance. "
                f"Got {type(estimator).__name__}."
            )

        param_combinations = estimator.cv_results_["params"]

        edge_params = []
        for param_name, best_value in estimator.best_params_.items():
            tried = [
                param_combination[param_name]
                for param_combination in param_combinations
                if param_name in param_combination
            ]
            if len(set(tried)) < 2 or not all(
                isinstance(value, numbers.Real)
                and not isinstance(value, bool | np.bool_)
                for value in tried
            ):
                continue
            search_low, search_high = min(tried), max(tried)
            if not isinstance(best_value, numbers.Real) or isinstance(
                best_value, bool | np.bool_
            ):
                continue
            if float(best_value) == float(search_low):
                space_low = _get_space_bound(
                    estimator.estimator, param_name=param_name, side="left"
                )
                if space_low is not None and float(search_low) == space_low:
                    continue
                edge_params.append((param_name, "minimum"))
            elif float(best_value) == float(search_high):
                space_high = _get_space_bound(
                    estimator.estimator, param_name=param_name, side="right"
                )
                if space_high is not None and float(search_high) == space_high:
                    continue
                edge_params.append((param_name, "maximum"))

        if not edge_params:
            return None
        details = ", ".join(f"{name} ({bound})" for name, bound in edge_params)
        return (
            f"{len(edge_params)} hyperparameter(s) are on the edge of the explored "
            f"search space: {details}. Consider extending the search range or "
            "increasing the number of iterations for randomized search."
        )
