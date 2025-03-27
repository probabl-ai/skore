from time import perf_counter


class MeasureTime:
    """Measure the time to go through the context, in seconds.

    Examples
    --------
    >>> from time import sleep
    >>> with MeasureTime() as time_taken:
    ...     sleep(0.01)
    >>> assert 0 <= time_taken() <= 1 # Note: time_taken is a callable
    """

    def __enter__(self):
        self.start = perf_counter()
        self.end = self.start
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = perf_counter()

    def __call__(self):
        return self.end - self.start
