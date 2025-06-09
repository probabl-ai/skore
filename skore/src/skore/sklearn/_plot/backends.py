"""Backend configuration for plotting in skore.

This module provides a Pandas-style plotting backend system that allows users
to globally configure which plotting library (matplotlib or plotly) to use
for all visualizations in skore.
"""

import importlib.util
import warnings
from typing import Literal, Optional

# Supported backends
SUPPORTED_BACKENDS = ("matplotlib", "plotly")

# Default backend
_DEFAULT_BACKEND = "matplotlib"

# Current active backend
_CURRENT_BACKEND = _DEFAULT_BACKEND

# Backend availability cache
_BACKEND_AVAILABILITY_CACHE: Optional[dict[str, bool]] = None


def get_backend() -> str:
    """Get the currently active plotting backend.

    Returns
    -------
    str
        Name of the current backend ("matplotlib" or "plotly").

    Examples
    --------
    >>> import skore
    >>> skore.get_backend()
    'matplotlib'
    >>> skore.set_backend("plotly")
    >>> skore.get_backend()
    'plotly'
    """
    return _CURRENT_BACKEND


def set_backend(backend: Literal["matplotlib", "plotly"]) -> None:
    """Set the current plotting backend globally.

    This affects all new plots created after calling this function.
    Individual plots can still override the backend using the `backend` parameter.

    Parameters
    ----------
    backend : {"matplotlib", "plotly"}
        The backend to use for plotting.

    Raises
    ------
    ValueError
        If the specified backend is not supported.
    ImportError
        If the backend is not available (e.g., plotly not installed).

    Examples
    --------
    >>> import skore
    >>> skore.set_backend("plotly")  # All future plots use Plotly
    >>> # Create a report and plot - will use Plotly by default
    >>> report = skore.EstimatorReport(estimator, **data)
    >>> display = report.metrics.precision_recall()
    >>> display.plot()  # Uses Plotly
    >>>
    >>> # Individual plots can still override
    >>> display.plot(backend="matplotlib")  # Uses Matplotlib
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' is not supported. "
            f"Supported backends are: {', '.join(SUPPORTED_BACKENDS)}"
        )

    # Check if backend is available
    if not _check_backend_available(backend):
        if backend == "plotly":
            raise ImportError(
                "Plotly is required for plotly backend. "
                "Install it with: pip install plotly"
            )
        elif backend == "matplotlib":
            raise ImportError(
                "Matplotlib is required for matplotlib backend. "
                "Install it with: pip install matplotlib"
            )

    global _CURRENT_BACKEND
    _CURRENT_BACKEND = backend


def _check_backend_available(backend: str) -> bool:
    """Check if a backend is available (with caching).

    Parameters
    ----------
    backend : str
        The name of the backend to check.

    Returns
    -------
    bool
        True if the backend is available, False otherwise.
    """
    global _BACKEND_AVAILABILITY_CACHE

    # Initialize cache if needed
    if _BACKEND_AVAILABILITY_CACHE is None:
        _BACKEND_AVAILABILITY_CACHE = {}

    # Return cached result if available
    if backend in _BACKEND_AVAILABILITY_CACHE:
        return _BACKEND_AVAILABILITY_CACHE[backend]

    # Check availability using importlib.util.find_spec and cache result
    is_available = False
    if backend == "matplotlib":
        is_available = (
            importlib.util.find_spec("matplotlib") is not None
            and importlib.util.find_spec("matplotlib.pyplot") is not None
        )
    elif backend == "plotly":
        is_available = (
            importlib.util.find_spec("plotly") is not None
            and importlib.util.find_spec("plotly.graph_objects") is not None
            and importlib.util.find_spec("plotly.express") is not None
            and importlib.util.find_spec("plotly.subplots") is not None
        )

    _BACKEND_AVAILABILITY_CACHE[backend] = is_available
    return is_available


def list_backends() -> tuple[str, ...]:
    """List all supported backends.

    Returns
    -------
    tuple of str
        Tuple of supported backend names.

    Examples
    --------
    >>> import skore
    >>> skore.list_backends()
    ('matplotlib', 'plotly')
    """
    return SUPPORTED_BACKENDS


def reset_backend() -> None:
    """Reset the backend to the default (matplotlib).

    Examples
    --------
    >>> import skore
    >>> skore.set_backend("plotly")
    >>> skore.get_backend()
    'plotly'
    >>> skore.reset_backend()
    >>> skore.get_backend()
    'matplotlib'
    """
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = _DEFAULT_BACKEND


def get_available_backends() -> dict[str, bool]:
    """Get availability status of all backends.

    Returns
    -------
    dict
        Dictionary mapping backend names to their availability status.

    Examples
    --------
    >>> import skore
    >>> skore.get_available_backends()
    {'matplotlib': True, 'plotly': True}  # if both are installed
    >>> # Or if plotly not installed:
    >>> skore.get_available_backends()
    {'matplotlib': True, 'plotly': False}
    """
    return {
        backend: _check_backend_available(backend) for backend in SUPPORTED_BACKENDS
    }


def validate_backend(backend: Optional[str]) -> str:
    """Validate and return a backend, using global default if None.

    This is a utility function for internal use by plotting methods.

    Parameters
    ----------
    backend : str or None
        The backend to validate. If None, returns the global backend.

    Returns
    -------
    str
        Valid backend name.

    Raises
    ------
    ValueError
        If the backend is not supported.
    ImportError
        If the backend is not available.
    """
    if backend is None:
        backend = get_backend()

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' is not supported. "
            f"Supported backends are: {', '.join(SUPPORTED_BACKENDS)}"
        )

    if not _check_backend_available(backend):
        if backend == "plotly":
            raise ImportError(
                "Plotly is required for plotly backend. "
                "Install it with: pip install plotly"
            )
        elif backend == "matplotlib":
            raise ImportError(
                "Matplotlib is required for matplotlib backend. "
                "Install it with: pip install matplotlib"
            )

    return backend


def _clear_cache() -> None:
    """Clear the backend availability cache.

    This is mainly for testing purposes.
    """
    global _BACKEND_AVAILABILITY_CACHE
    _BACKEND_AVAILABILITY_CACHE = None


def _warn_if_backend_unavailable(backend: str) -> None:
    """Issue a warning if the specified backend is not available.

    Parameters
    ----------
    backend : str
        The backend to check.
    """
    if not _check_backend_available(backend):
        warnings.warn(
            f"Backend '{backend}' is not available. "
            f"Please install the required dependencies.",
            UserWarning,
            stacklevel=3,
        )


# Initialize the backend availability cache on import
get_available_backends()
