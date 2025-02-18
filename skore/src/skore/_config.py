"""Global configuration state and functions for management."""

import threading
import time
from contextlib import contextmanager

_global_config = {
    "show_progress": True,
}
_threadlocal = threading.local()


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration.

    If the configuration does not exist, copy the default global configuration.
    """
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for local skore configuration.
    set_config : Set global skore configuration.

    Examples
    --------
    >>> import skore
    >>> config = skore.get_config()
    >>> config.keys()
    dict_keys([...])
    """
    # Return a copy of the threadlocal configuration so that users will
    # not be able to modify the configuration with the returned dict.
    return _get_threadlocal_config().copy()


def set_config(
    show_progress: bool = None,
):
    """Set global skore configuration.

    Parameters
    ----------
    show_progress : bool, default=None
        If True, show progress bars. Otherwise, do not show them.

    See Also
    --------
    config_context : Context manager for local skore configuration.
    get_config : Retrieve current values of the global configuration.

    Examples
    --------
    >>> from skore import set_config
    >>> set_config(show_progress=False)  # doctest: +SKIP
    """
    local_config = _get_threadlocal_config()

    if show_progress is not None:
        local_config["show_progress"] = show_progress


@contextmanager
def config_context(
    *,
    show_progress: bool = None,
):
    """Context manager for local skore configuration.

    Parameters
    ----------
    show_progress : bool, default=None
        If True, show progress bars. Otherwise, do not show them.

    Yields
    ------
    None.

    See Also
    --------
    set_config : Set global skore configuration.
    get_config : Retrieve current values of the global configuration.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    Examples
    --------
    >>> import skore
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from skore import CrossValidationReport
    >>> with skore.config_context(show_progress=False):
    ...     X, y = make_classification(random_state=42)
    ...     estimator = LogisticRegression()
    ...     report = CrossValidationReport(estimator, X=X, y=y, cv_splitter=2)
    """
    old_config = get_config()
    set_config(
        show_progress=show_progress,
    )

    try:
        yield
    finally:
        set_config(**old_config)


def _set_show_progress_for_testing(show_progress, sleep_duration):
    """Set the value of show_progress for testing purposes after some waiting.

    This function should exist in a Python module rather than in tests, otherwise
    joblib will not be able to pickle it.
    """
    with config_context(show_progress=show_progress):
        time.sleep(sleep_duration)
        return get_config()["show_progress"]
