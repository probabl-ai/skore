"""Global configuration state and functions for management."""

from __future__ import annotations

from contextlib import contextmanager
from threading import current_thread, main_thread
from threading import local as Local


class LocalConfiguration(Local):
    def __init__(
        self,
        *,
        show_progress=True,
        plot_backend="matplotlib",
        ignore_checks: list[str] | tuple[str, ...] | None = None,
    ):
        self.show_progress = show_progress
        self.plot_backend = plot_backend
        self.ignore_checks = ignore_checks


class Configuration:
    """Configuration for `skore` behavior.

    You can read and set options via attribute access. In parallel processing (e.g.
    ``joblib.Parallel``), each thread receives its own copy of the configuration;
    changing attributes inside a worker thread only affects that thread and does not
    modify the global configuration in the main thread.

    Attributes
    ----------
    show_progress : bool
        Whether to show progress bars for long-running operations.
        Default is ``True`` (or ``False`` when joblib < 1.4).

    plot_backend : str
        Backend used for rendering plots (e.g. ``"matplotlib"``).
        Default is ``"matplotlib"``.

    ignore_checks : list of str or tuple of str or None
        Global diagnostic codes ignored by ``report.diagnosis.summarize(...)``.
        Default is ``None``.

    Examples
    --------
    **Global configuration** using the ``configuration`` instance from skore:

    >>> # xdoctest: +SKIP
    >>> from skore import configuration
    >>> configuration.show_progress = False
    >>> configuration.plot_backend = "matplotlib"
    >>> configuration.ignore_checks = ["SKD001"]

    **Temporary overrides** using the context manager (previous values are
    restored on exit):

    >>> # xdoctest: +SKIP
    >>> with configuration(show_progress=False):
    ...     report.fit(X, y)
    >>> with configuration(plot_backend="plotly"):
    ...     report.plot()
    >>> with configuration(ignore_checks=["SKD002"]):
    ...     report.diagnosis.summarize()
    """

    def __init__(self):
        self.local = LocalConfiguration()

    def __repr__(self):
        return (
            f"Configuration("
            f"show_progress={self.local.show_progress}, "
            f"plot_backend={self.local.plot_backend!r}, "
            f"ignore_checks={self.local.ignore_checks}"
            ")"
        )

    @property
    def show_progress(self):
        return self.local.show_progress

    @show_progress.setter
    def show_progress(self, value):
        if current_thread().ident != main_thread().ident:
            self.local.show_progress = value
            return

        self.local = LocalConfiguration(
            show_progress=value,
            plot_backend=self.local.plot_backend,
            ignore_checks=self.local.ignore_checks,
        )

    @property
    def plot_backend(self):
        return self.local.plot_backend

    @plot_backend.setter
    def plot_backend(self, value):
        if current_thread().ident != main_thread().ident:
            self.local.plot_backend = value
            return

        self.local = LocalConfiguration(
            show_progress=self.local.show_progress,
            plot_backend=value,
            ignore_checks=self.local.ignore_checks,
        )

    @property
    def ignore_checks(self):
        return self.local.ignore_checks

    @ignore_checks.setter
    def ignore_checks(self, value):
        if current_thread().ident != main_thread().ident:
            self.local.ignore_checks = value
            return

        self.local = LocalConfiguration(
            show_progress=self.local.show_progress,
            plot_backend=self.local.plot_backend,
            ignore_checks=value,
        )

    @contextmanager
    def __call__(
        self,
        *,
        show_progress=...,
        plot_backend=...,
        ignore_checks=...,
    ):
        show_progress_copy = self.show_progress
        plot_backend_copy = self.plot_backend
        ignore_checks_copy = self.ignore_checks

        if show_progress is not ...:
            self.show_progress = show_progress

        if plot_backend is not ...:
            self.plot_backend = plot_backend

        if ignore_checks is not ...:
            self.ignore_checks = ignore_checks

        try:
            yield
        finally:
            self.show_progress = show_progress_copy
            self.plot_backend = plot_backend_copy
            self.ignore_checks = ignore_checks_copy


configuration = Configuration()
