from datetime import datetime, timezone

import pytest
from skore.persistence.repository import ItemRepository, ViewRepository
from skore.persistence.storage import InMemoryStorage
from skore.project import Project


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
