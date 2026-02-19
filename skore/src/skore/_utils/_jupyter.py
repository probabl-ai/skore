"""Utilities for optional Jupyter/IPython and ipywidgets support."""

__all__ = ["_jupyter_dependencies_available"]


def _jupyter_dependencies_available() -> bool:
    """Return True if IPython and ipywidgets are importable (optional Jupyter deps)."""
    try:
        import IPython.display  # noqa: F401
        import ipywidgets  # noqa: F401

        return True
    except ImportError:
        return False
