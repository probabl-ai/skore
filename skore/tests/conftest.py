from datetime import datetime, timezone

import pytest


def pytest_configure(config):
    # Use matplotlib agg backend during the tests including doctests
    import matplotlib

    matplotlib.use("agg")


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


@pytest.fixture
def mock_now():
    return datetime.now(tz=timezone.utc)


@pytest.fixture
def mock_nowstr(mock_now):
    return mock_now.isoformat()


@pytest.fixture
def MockDatetime(mock_now):
    class MockDatetime:
        def __init__(self, *args, **kwargs): ...

        @staticmethod
        def now(*args, **kwargs):
            return mock_now

    return MockDatetime


@pytest.fixture(scope="function")
def pyplot():
    """Setup and teardown fixture for matplotlib.

    This fixture closes the figures before and after running the functions.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    from matplotlib import pyplot

    pyplot.close("all")
    yield pyplot
    pyplot.close("all")
