from datetime import datetime, timezone

import pytest
from skore.item.item_repository import ItemRepository
from skore.persistence.in_memory_storage import InMemoryStorage
from skore.project import Project
from skore.view.view_repository import ViewRepository


def pytest_configure(config):
    # Use matplotlib agg backend during the tests including doctests
    try:
        import matplotlib

        matplotlib.use("agg")
    except ImportError:
        pass


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
        item_repository=item_repository,
        view_repository=view_repository,
    )


@pytest.fixture(scope="function")
def pyplot():
    """Setup and teardown fixture for matplotlib.

    This fixture checks if we can import matplotlib. If not, the tests will be
    skipped. Otherwise, we close the figures before and after running the
    functions.

    Returns
    -------
    pyplot : module
        The ``matplotlib.pyplot`` module.
    """
    pyplot = pytest.importorskip("matplotlib.pyplot")
    pyplot.close("all")
    yield pyplot
    pyplot.close("all")
