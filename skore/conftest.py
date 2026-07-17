"""Top-level conftest applying to both `tests/` and doctests under `src/`."""

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
