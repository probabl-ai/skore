from datetime import datetime, timezone

import pytest
from skore.persistence.repository import ItemRepository
from skore.persistence.storage import InMemoryStorage
from skore.project.project import Project


def pytest_configure(config):
    # Use matplotlib agg backend during the tests including doctests
    import matplotlib

    matplotlib.use("agg")


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


@pytest.fixture
def in_memory_project(monkeypatch):
    monkeypatch.delattr("skore.project.Project.__init__")

    project = Project()
    project.path = None
    project.name = "test"
    project._item_repository = ItemRepository(storage=InMemoryStorage())
    project._storage_initialized = True

    return project


@pytest.fixture
def on_disk_project(tmp_path):
    return Project(tmp_path / "project.skore")


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


@pytest.fixture(autouse=True)
def monkeypatch_tmpdir(monkeypatch, tmp_path):
    """A pytest fixture that temporarily changes the system's temporary directory.

    This fixture modifies the TMPDIR environment variable to point to a temporary
    path provided by pytest's tmp_path fixture. It also handles the cached nature
    of tempfile.gettempdir() by reloading the tempfile module.
    """
    import tempfile
    from importlib import reload

    with monkeypatch.context() as mp:
        mp.setenv("TMPDIR", str(tmp_path))

        # The first call of `tempfile.gettempdir` being cached, reload the module.
        # https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
        reload(tempfile)

        yield
