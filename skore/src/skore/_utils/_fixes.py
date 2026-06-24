from typing import Any

import joblib

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
