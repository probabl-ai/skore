from collections.abc import Iterable
from functools import partial
from operator import length_hint
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

    def __init__(self, description: str, total: float | None):
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
    def description(self) -> str:
        """Description of the progress bar."""
        return self._description

    @description.setter
    def description(self, value: str):
        """Set description of the progress bar."""
        self._description = value
        self._progress.update(self._task, description=value, refresh=True)

    @property
    def total(self) -> float | None:
        """Total number of steps before the progress bar is considered completed."""
        return self._total

    @total.setter
    def total(self, value: float):
        """Set total number of steps before the progress bar is considered completed."""
        self._total = value
        self._progress.update(self._task, total=value, refresh=True)

    def advance(self):
        """Advance the progress bar by one step."""
        self._progress.update(self._task, advance=1, refresh=True)


def track(
    sequence: Iterable[Any], description: str, total: float | None = None
) -> Iterable[Any]:
    """Track progress by iterating over a sequence."""
    if total is None:
        total = float(length_hint(sequence)) or None

    with ProgressBar(description=description, total=total) as progress:
        for value in sequence:
            yield value
            progress.advance()
