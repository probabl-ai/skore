"""Top-level conftest applying to both `tests/` and doctests under `src/`."""

# ruff: noqa: E402
import os

# Cap native thread pools before NumPy and sklearn are imported below, otherwise
# pytest-xdist workers oversubscribe the CPUs (BLAS and joblib nesting inside each
# worker) and the suite gets slower than it is serially, especially on Windows.
for _thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ[_thread_env_var] = "1"

import matplotlib
import matplotlib.pyplot

import skore


def pytest_configure(config):
    """Set up global test configuration.

    Some of these could be set in fixtures, but doctests do not run fixtures.
    """
    matplotlib.use("agg")

    # Disable progress bars during tests to avoid rich interfering with
    # doctest stdout capture.
    skore.configuration.show_progress = False


def pytest_runtest_teardown(item):
    """Close any matplotlib figures left open by the test.

    Guards against silent figure accumulation across the suite (which used to
    trip matplotlib's `figure.max_open_warning`). Applied via a hook rather
    than an autouse fixture so it also covers doctests.
    """
    matplotlib.pyplot.close("all")
