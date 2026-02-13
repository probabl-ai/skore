"""Customizations of :mod:`joblib` and :mod:`threadpoolctl` tools for skore usage."""

from functools import update_wrapper, wraps

from skore import configuration
from skore._config import LocalConfiguration


def delayed(function):
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
        configuration.local = LocalConfiguration(**self.local)
        return self.function(*args, **kwargs)
