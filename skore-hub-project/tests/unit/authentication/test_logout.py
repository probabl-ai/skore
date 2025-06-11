from datetime import datetime, timezone

from skore_hub_project.authentication import token as Token
from skore_hub_project.authentication.logout import logout

DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()


def test_logout():
    Token.save("A", "B", DATETIME_MAX)

    assert Token.exists()
    assert Token.access(refresh=False) == "A"

    logout()

    assert not Token.exists()
