"""Customizations of :mod:`joblib` and :mod:`threadpoolctl` tools for skore usage."""

from functools import update_wrapper, wraps

from skore import configuration
from skore._config import LocalConfiguration


def delayed(function):
    """Capture the arguments of a function to delay its execution.

    This alternative to ``joblib.delayed`` is meant to be used in conjunction
    with ``joblib.Parallel``.

    It ensures each task has its own copy of the ``skore`` configuration.

    Parameters
    ----------
    function : callable
        The function to be delayed.

    Returns
    -------
    output: tuple
        Tuple containing the delayed function, the positional arguments, and the
        keyword arguments.
    """
    local = configuration.local.__dict__.copy()

    @wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function, local), args, kwargs

    return delayed_function


class _FuncWrapper:
    def __init__(self, function, local):
        self.function = function
        self.local = local

        update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        # `joblib.Parallel` is capable of reusing a process/thread for multiple tasks,
        # so we must reset the configuration to its initial state before executing each
        # task.
        #
        # This way, we ensure that there is no side-effect between tasks executed on the
        # same computing unit.
        configuration.local = LocalConfiguration(**self.local)
        return self.function(*args, **kwargs)
