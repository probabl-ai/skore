from typing import Any

import joblib
from sklearn.utils.validation import check_is_fitted

from skore._externals._sklearn_compat import parse_version


# FIXME: Remove this function once we support only joblib >= 1.4
def _validate_joblib_parallel_params(**kwargs: Any) -> dict[str, Any]:
    """Validate the parameters for `joblib.Parallel`.

    Currently this function is in charge of removing the parameter `return_as`
    because it is not supported by joblib < 1.4.
    """
    joblib_version = parse_version(joblib.__version__)
    if joblib_version < parse_version("1.4"):
        kwargs.pop("return_as", None)
    return kwargs


def skore_check_is_fitted(estimator):
    """More thorough version of sklearn's `check_is_fitted` covering skorch models.

    Skorch models that are not fitted pass through the sklearn `check_is_fitted`
    when they are not fitted. Skorch models have three attributes that exist on not
    fitted models which do not exist on other models. This function checks if the
    estimator has these three attributes to determine if the estimator is a skorch
    model. If so, extra attributes are passed to sklearn's `check_is_fitted`
    to avoid silently passing not fitted skorch models to `EstimatorReport`s.
    """
    if not all(
        attr in vars(estimator)
        for attr in ["history_", "initialized_", "virtual_params_"]
    ):
        check_is_fitted(estimator)
    else:  # skorch models
        check_is_fitted(
            estimator,
            attributes=[
                "init_context_",
                "callbacks_",
                "prefixes_",
                "cuda_dependent_attributes_",
                "module_",
                "criterion_",
                "optimizer_",
                "classes_inferred_",
            ],
        )
