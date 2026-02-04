from collections.abc import Callable
from functools import wraps, partial
from threading import RLock, get_ident
from typing import Any, TypeVar

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from skore._config import get_config

T = TypeVar("T")

SkinnedProgress = partial(
    Progress,
    SpinnerColumn(),
    TextColumn("[bold cyan]{task.description}"),
    BarColumn(
        complete_style="dark_orange",
        finished_style="dark_orange",
        pulse_style="orange1",
    ),
    TextColumn("[orange1]{task.percentage:>3.0f}%"),
    expand=False,
    # transient=True,
    disable=(not get_config()["show_progress"]),
)


class ProgressBar:
    def __init__(self, description):
        self.__description = description
        self.__total = None
        self.__progress = SkinnedProgress()
        self.__task = self.__progress.add_task(description, total=None)

    def __enter__(self):
        self.__progress.start()
        return self

    def __exit__(self, type, value, traceback):
        self.__progress.stop()

    @property
    def description(self) -> str | None:
        return self.__description

    @description.setter
    def description(self, value: str | None):
        self.__progress.update(self.__task, description=value, refresh=True)

    @property
    def total(self) -> float:
        return self.__total

    @total.setter
    def total(self, value: float):
        self.__total = value
        self.__progress.update(self.__task, total=value, refresh=True)

    def advance(self):
        self.__progress.update(self.__task, advance=1, refresh=True)


# class ProgressBarPool:
#     def __init__(self):
#         self.__lock = RLock()
#         self.__pool = {}

#     def __setitem__(self, description, progress_bar):
#         with self.__lock:
#             self.__pool[(get_ident(), description)] = progress_bar

#     def __getitem__(self, description):
#         with self.__lock:
#             return self.__pool[(get_ident(), description)]

#     def __delitem__(self, description):
#         with self.__lock:
#             del self.__pool[(get_ident(), description)]

#     def __contains__(self, description):
#         with self.__lock:
#             return (get_ident(), description) in self.__pool


def progress_decorator(
    describe: str | Callable[..., str],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate class methods to add a progress bar.

    This decorator adds a Rich progress bar to class methods, displaying progress
    during execution. The progress bar automatically disappears after completion.

    Parameters
    ----------
    description : str or callable
        The description of the progress bar. If a callable, it should take the
        self object as an argument and return a string.

    Returns
    -------
    decorator : function
        A decorator that wraps the input function and adds a progress bar to it.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # func.describe = (
        #     description if callable(description) else (lambda _: description)
        # )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            report = args[0]

            if hasattr(report, "_parent"):  # report is an accessor
                report = report._parent

            description = describe(report) if callable(describe) else describe

            with ProgressBar(description) as progress:
                return func(*args, **kwargs, progress=progress)

            # assert description not in self._progress_pool, "Something wrong"
            # self._progress_pool[description] = progress_bar

            # the problem:
            # in multithread, 2 threads can work on the same progress
            # the first will delete the progress, before the second finishes
            # -> thread-safety issue
            # the solutions:
            # - either we create one progress per thread
            #   https://docs.python.org/3/library/threading.html#threading.get_ident
            # - or we don't delete the progress here, but we delete it in __reduce__
            # - or we don't do anything when show_progress=False

            # try:
            #     ...
            # finally:
            #     progress_bar.stop()
            #     del self._progress_pool[description]

        return wrapper

    return decorator
