from datetime import datetime, timezone

from pytest import raises
from skore_hub_project.authentication.logout import logout
from skore_hub_project.authentication.token import Token

DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()


def test_logout():
    Token.save("A", "B", DATETIME_MAX)

    assert Token.exists()
    assert Token.access(refresh=False) == "A"

    logout()

    assert not Token.exists()
