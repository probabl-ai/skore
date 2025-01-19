from datetime import datetime, timezone

import pytest
import skore
from skore.item.item_repository import ItemRepository
from skore.persistence.in_memory_storage import InMemoryStorage
from skore.project import Project
from skore.view.view_repository import ViewRepository


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
def in_memory_project():
    item_repository = ItemRepository(storage=InMemoryStorage())
    view_repository = ViewRepository(storage=InMemoryStorage())
    return Project(
        name="in_memory_project.skore",
        item_repository=item_repository,
        view_repository=view_repository,
    )


@pytest.fixture
def on_disk_project(tmp_path):
    project = skore.open(tmp_path / "project")
    return project


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
