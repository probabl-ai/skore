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
    ClassName,
    cast_report,
    collapse_equivalents,
    get_fitted_estimator,
)
from skore._utils.skrub import (
    find_estimators,
    is_skrub_learner,
    is_tunable,
    iter_fitted_estimator_steps,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


def skrub_classes_with_tunable_recommended_params(estimator) -> set[ClassName]:
    """Return estimator class names with skrub choices on recommended params."""
    tuned_classes: set[ClassName] = set()
    for unfitted in find_estimators(estimator.data_op, include_nested=False):
        estimators = (
            [step for _, step in unfitted.steps]
            if isinstance(unfitted, Pipeline)
            else [unfitted]
        )
        for est in estimators:
            class_name = type(est).__name__
            if class_name not in HYPERPARAMETERS_TO_TUNE:
                continue
            for param_name in HYPERPARAMETERS_TO_TUNE[class_name]:
                if param_name in INFRASTRUCTURE_PARAMS:
                    continue
                if is_tunable(getattr(est, param_name, None)):
                    tuned_classes.add(class_name)
    return tuned_classes


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

        candidates = [
            (class_name, step)
            for class_name, step in iter_fitted_estimator_steps(estimator)
            if class_name in HYPERPARAMETERS_TO_TUNE
        ]
        if not candidates:
            raise CheckNotApplicable("No parameter to recommend.")

        skrub_tuned_classes = (
            skrub_classes_with_tunable_recommended_params(estimator)
            if is_skrub_learner(estimator)
            else set()
        )

        messages: list[str] = []
        for class_name, step in candidates:
            if class_name in skrub_tuned_classes:
                continue
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
