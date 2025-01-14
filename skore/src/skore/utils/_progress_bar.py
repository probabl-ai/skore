from functools import wraps

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)


def progress_decorator(description):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self_obj = args[0]

            desc = description(self_obj) if callable(description) else description

            if getattr(self_obj, "_parent_progress", None) is not None:
                progress = self_obj._parent_progress
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
                )
                progress.start()

            task = progress.add_task(desc, total=None)
            self_obj._progress_info = {
                "current_progress": progress,
                "current_task": task,
            }
            has_errored = False
            try:
                result = func(*args, **kwargs)
                if progress.tasks[task].total is not None:
                    progress.update(
                        task, completed=progress.tasks[task].total, refresh=True
                    )
                return result
            except Exception:
                has_errored = True
                raise
            finally:
                if self_obj._parent_progress is None:
                    if not has_errored:
                        progress.update(
                            task, completed=progress.tasks[task].total, refresh=True
                        )
                    progress.stop()

        return wrapper

    return decorator
