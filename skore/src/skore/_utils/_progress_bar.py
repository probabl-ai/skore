from collections.abc import Callable
from functools import partial, wraps
from typing import Any, TypeVar

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from skore._config import get_config

T = TypeVar("T")
Description = str | Callable[..., str]
Function = Callable[..., T]
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
    transient=True,
    disable=(not get_config()["show_progress"]),
)


class ProgressBar:
    def __init__(self, description=None, total=None):
        self.__description = description
        self.__total = total
        self.__progress = SkinnedProgress()
        self.__task = self.__progress.add_task(description, total=total)

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
    def total(self, value: float | None):
        self.__total = value
        self.__progress.update(self.__task, total=value, refresh=True)

    def advance(self):
        self.__progress.update(self.__task, advance=1, refresh=True)


def progress_decorator(description: Description) -> Callable[[Function], Function]:
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
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            report = args[0]._parent if hasattr(args[0], "_parent") else args[0]
            task = description(report) if callable(description) else description

            with ProgressBar(task) as progress:
                return func(*args, **kwargs, progress=progress)

        return wrapper

    return decorator
