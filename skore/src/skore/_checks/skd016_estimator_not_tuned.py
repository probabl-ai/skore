from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils._pprint import _changed_params

from skore._checks.base import Check
from skore._checks.tunable_hyperparameters import (
    HYPERPARAMETERS_TO_TUNE,
    INFRASTRUCTURE_PARAMS,
)
from skore._checks.utils import (
    CheckNotApplicable,
    cast_report,
    collapse_equivalents,
    get_fitted_estimator,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckEstimatorNotTuned(Check):
    """Check that the estimator has at least some non-default hyperparameters (SKD016).

    Fires when every parameter of the estimator (or, for pipelines, of every
    step whose class is in the recommendation table) is at scikit-learn's
    default value, ignoring infrastructure params (random_state, n_jobs, ...).
    Suggests the recommended tuning axes from ``HYPERPARAMETERS_TO_TUNE``.

    Skipped (:class:`CheckNotApplicable`) when the estimator is a
    :class:`~sklearn.model_selection.BaseSearchCV` instance, since SKD015
    covers that case.
    """

    code = "SKD016"
    title = "Estimator not tuned"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd016-estimator-not-tuned"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        report = cast_report(report)
        estimator = get_fitted_estimator(report)
        if isinstance(estimator, BaseSearchCV):
            raise CheckNotApplicable("Estimator is a BaseSearchCV instance.")

        if isinstance(estimator, Pipeline):
            candidates = [
                (type(step).__name__, step)
                for _, step in estimator.steps
                if type(step).__name__ in HYPERPARAMETERS_TO_TUNE
            ]
            if not candidates:
                raise CheckNotApplicable(
                    "No parameter to recommend for any of the steps."
                )
        else:
            class_name = type(estimator).__name__
            if class_name not in HYPERPARAMETERS_TO_TUNE:
                raise CheckNotApplicable("No parameter to recommend for the estimator.")
            candidates = [(class_name, estimator)]

        messages: list[str] = []
        for class_name, step in candidates:
            if set(_changed_params(step)) - INFRASTRUCTURE_PARAMS:
                continue
            recommended = collapse_equivalents(
                HYPERPARAMETERS_TO_TUNE[class_name], set()
            )
            messages.append(f"{sorted(recommended)} for {class_name}")

        if not messages:
            return None
        messages.sort()
        return (
            "Estimator(s) left at default settings; consider tuning: "
            f"{'; '.join(sorted(messages))}."
        )
