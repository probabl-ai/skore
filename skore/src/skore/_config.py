"""Global configuration state and functions for management."""

from __future__ import annotations

from contextlib import contextmanager
from threading import current_thread, main_thread
from threading import local as Local


class LocalConfiguration(Local):
    show_progress = True
    plot_backend = "matplotlib"

    def __init__(self, /, **kw):
        self.__dict__.update(kw)


class Configuration:
    local = LocalConfiguration()

    def __repr__(self):
        return (
            f"Configuration("
            f"show_progress={Configuration.local.show_progress}, "
            f"plot_backend={Configuration.local.plot_backend!r}"
            ")"
        )

    @property
    def show_progress(self):
        return Configuration.local.show_progress

    @show_progress.setter
    def show_progress(self, value):
        if current_thread().ident != main_thread().ident:
            Configuration.local.show_progress = value
            return

        Configuration.local = LocalConfiguration(
            show_progress=value,
            plot_backend=Configuration.local.plot_backend,
        )

    @property
    def plot_backend(self):
        return Configuration.local.plot_backend

    @plot_backend.setter
    def plot_backend(self, value):
        if current_thread().ident != main_thread().ident:
            Configuration.local.plot_backend = value
            return

        Configuration.local = LocalConfiguration(
            show_progress=Configuration.local.show_progress,
            plot_backend=value,
        )

    @contextmanager
    def context(self, *, show_progress=..., plot_backend=...):
        show_progress_copy = Configuration.show_progress
        plot_backend_copy = Configuration.plot_backend

        if show_progress is not ...:
            self.show_progress = show_progress

        if plot_backend is not ...:
            self.plot_backend = plot_backend

        try:
            yield
        finally:
            self.show_progress = show_progress_copy
            self.plot_backend = plot_backend_copy


configuration = Configuration()
