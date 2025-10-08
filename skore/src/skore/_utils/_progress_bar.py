from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

from skore._config import get_config

T = TypeVar("T")
DescriptionType = str | Callable[..., str]


def progress_decorator(
    description: DescriptionType,
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
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            self_obj: Any = args[0]

            if hasattr(self_obj, "_parent"):
                # self_obj is an accessor
                self_obj = self_obj._parent

            desc = description(self_obj) if callable(description) else description

            created_progress = False

            if getattr(self_obj, "_progress_info", None) is not None:
                progress = self_obj._progress_info["current_progress"]
            else:
                progress = Progress(
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
                    disable=not get_config()["show_progress"],
                )
                progress.start()
                created_progress = True

            # assigning progress to child reports
            reports_to_cleanup: list[Any] = []
            if hasattr(self_obj, "reports_"):
                for report in self_obj.reports_.values():
                    if hasattr(report, "_progress_info"):
                        report._progress_info = {"current_progress": progress}
                        reports_to_cleanup.append(report)

            task = progress.add_task(desc, total=None)
            self_obj._progress_info = {
                "current_progress": progress,
                "current_task": task,
            }
            has_errored = False
            try:
                result = func(*args, **kwargs)
                progress.update(
                    task, completed=progress.tasks[task].total, refresh=True
                )
                return result
            except Exception:
                has_errored = True
                raise
            finally:
                if created_progress:
                    if not has_errored:
                        progress.update(
                            task, completed=progress.tasks[task].total, refresh=True
                        )
                    progress.stop()

                # clean up child reports
                for report in reports_to_cleanup:
                    report._progress_info = None

                # clean up to make object pickable
                self_obj._progress_info = None

        return wrapper

    return decorator
