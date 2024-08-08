from datetime import UTC, datetime

import pytest


@pytest.fixture
def mock_now():
    return datetime.now(tz=UTC)


@pytest.fixture
def mock_nowstr(mock_now):
    return mock_now.isoformat()
