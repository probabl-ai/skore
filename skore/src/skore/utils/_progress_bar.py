from functools import wraps

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)


def create_progress_bar():
    """Create a consistent progress bar style."""
    from skore import console  # avoid circular import

    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(
            complete_style="dark_orange",
            finished_style="dark_orange",
            pulse_style="orange1",
        ),
        TextColumn("[orange1]{task.percentage:>3.0f}%"),
        console=console,
        expand=False,
    )


class ProgressManager:
    _instance = None
    _progress = None

    @classmethod
    def get_progress(cls):
        if cls._progress is None:
            cls._progress = create_progress_bar()
            cls._progress.start()
        return cls._progress

    @classmethod
    def stop_progress(cls):
        if cls._progress is not None:
            try:
                cls._progress.stop()
            finally:
                cls._progress = None


class ProgressDecorator:
    def __init__(self, description):
        self.description = description

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self_obj = args[0]

            desc = (
                self.description(self_obj)
                if callable(self.description)
                else self.description
            )

            if getattr(self_obj, "_parent_progress", None) is not None:
                progress = self_obj._parent_progress
            else:
                progress = ProgressManager.get_progress()

            task = progress.add_task(desc, total=None)
            self_obj._progress_info = {
                "current_progress": progress,
                "current_task": task,
            }
            try:
                result = func(*args, **kwargs)
                if progress.tasks[task].total is not None:
                    progress.update(
                        task, completed=progress.tasks[task].total, refresh=True
                    )
                return result
            except Exception:
                raise

        return wrapper
