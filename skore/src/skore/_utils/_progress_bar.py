from collections.abc import Callable
from functools import partial, wraps
from inspect import signature
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
    transient=True,
    disable=(not get_config()["show_progress"]),
)


class ProgressBar:
    """Simplified progress bar based on ``rich.Progress``."""

    def __init__(self, description=None, total=None):
        self._description = description
        self._total = total
        self._progress = SkinnedProgress()
        self._task = self._progress.add_task(description, total=total)

    def __enter__(self):
        self._progress.start()
        return self

    def __exit__(self, type, value, traceback):
        self._progress.stop()

    @property
    def description(self) -> str | None:
        """Description of the progress bar."""
        return self._description

    @description.setter
    def description(self, value: str | None):
        """Set description of the progress bar."""
        self._description = value
        self._progress.update(self._task, description=value, refresh=True)

    @property
    def total(self) -> float:
        """Total number of steps before the progress bar is considered completed."""
        return self._total

    @total.setter
    def total(self, value: float | None):
        """Set total number of steps before the progress bar is considered completed."""
        self._total = value
        self._progress.update(self._task, total=value, refresh=True)

    def advance(self):
        """Advance the progress bar by one step."""
        self._progress.update(self._task, advance=1, refresh=True)


def progress_decorator(
    description: str | Callable[..., str],
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorate class methods to track progress.

    This decorator injects a ``ProgressBar`` object to the wrapped method, used to track
    progress during execution. The progress bar automatically disappears after
    completion.

    The wrapped method is responsible for advancing the progress bar and managing its
    size.

    Parameters
    ----------
    description : str or callable
        The description of the progress bar. If a callable, it should take the
        self object as an argument and return a string.

    Returns
    -------
    decorator : function
        A decorator that wraps the input function, and injects the progress bar as
        parameter.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        assert "progress" in signature(func).parameters, (
            "You can only decorate functions with `progress` parameter"
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            report = args[0]._parent if hasattr(args[0], "_parent") else args[0]
            task = description(report) if callable(description) else description

            with ProgressBar(task) as progress:
                return func(*args, **kwargs, progress=progress)

        return wrapper

    return decorator
