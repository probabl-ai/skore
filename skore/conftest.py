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
import pytest

import skore
from skore._config import LocalConfiguration


def pytest_configure(config):
    """Set up global test configuration.

    Some of these could be set in fixtures, but doctests do not run fixtures.
    """
    matplotlib.use("agg")

    # Disable progress bars during tests to avoid rich interfering with
    # doctest stdout capture.
    skore.configuration.show_progress = False


@pytest.fixture(autouse=True)
def monkeypatch_tmpdir(monkeypatch, tmp_path):
    """
    Change ``TMPDIR`` used by ``tempfile.gettempdir()`` to point to ``tmp_path``, so
    that it is automatically deleted after use, with no impact on user's environment.

    Force the reload of the ``tempfile`` module to change the cached return of
    ``tempfile.gettempdir()``.

    https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
    """
    import importlib
    import tempfile

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    importlib.reload(tempfile)


@pytest.fixture(autouse=True)
def monkeypatch_home(monkeypatch, tmp_path):
    """
    Change ``HOME`` used by ``os.path.expanduser()`` to point to ``tmp_path``, so
    that it is automatically deleted after use, with no impact on user's environment.

    https://docs.python.org/3/library/os.path.html#os.path.expanduser
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


@pytest.fixture(autouse=True)
def monkeypatch_keyring(monkeypatch):
    """
    Make the system keyring unavailable, so that tests can't read and write into the
    user's real keyring.

    Force a re-init of the keyring backend, ``keyring.get_keyring()`` caching the
    backend in ``keyring.core``.

    https://github.com/jaraco/keyring#disabling-keyring
    https://github.com/jaraco/keyring/blob/7603e7cadc254b4c6e3fc2b2f0916a005e78087d/keyring/core.py#L32
    """
    import keyring.core

    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    keyring.core.init_backend()


@pytest.fixture(autouse=True)
def monkeypatch_skore_hub_envars(monkeypatch):
    """
    Delete environment variables that can be used to reach the production's or user's
    HUB instance, disabling potential impacts on user's environment.
    """
    monkeypatch.delenv("SKORE_HUB_URI", raising=False)
    monkeypatch.delenv("SKORE_HUB_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def monkeypatch_configuration(monkeypatch):
    """Ensure that the test gets the default configuration,
    independently of the others."""
    monkeypatch.setattr("skore._config.configuration.local", LocalConfiguration())


def pytest_runtest_teardown(item):
    """Close any matplotlib figures left open by the test.

    Guards against silent figure accumulation across the suite (which used to
    trip matplotlib's `figure.max_open_warning`). Applied via a hook rather
    than an autouse fixture so it also covers doctests.
    """
    matplotlib.pyplot.close("all")
