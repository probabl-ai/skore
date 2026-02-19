"""Global configuration state and functions for management."""

from __future__ import annotations

from contextlib import contextmanager
from threading import current_thread, main_thread
from threading import local as Local


class LocalConfiguration(Local):
    def __init__(self, *, show_progress=True, plot_backend="matplotlib"):
        self.show_progress = show_progress
        self.plot_backend = plot_backend


class Configuration:
    def __init__(self):
        self.local = LocalConfiguration()

    def __repr__(self):
        return (
            f"Configuration("
            f"show_progress={self.local.show_progress}, "
            f"plot_backend={self.local.plot_backend!r}"
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
        )

    @contextmanager
    def __call__(self, *, show_progress=..., plot_backend=...):
        show_progress_copy = self.show_progress
        plot_backend_copy = self.plot_backend

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
