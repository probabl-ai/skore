from contextlib import contextmanager
from time import perf_counter


# https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time#69156219
@contextmanager
def _measure_time():
    """Measure the time to go through the context, in seconds.

    Returns
    -------
    Callable returning a float

    Examples
    --------
    >>> with _measure_time() as time_taken:
    ...     1+1
    >>> time_taken()  # Note: time_taken is a callable
    """
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
