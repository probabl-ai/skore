from datetime import datetime, timezone

from skore_hub_project.authentication import token, uri
from skore_hub_project.authentication.logout import logout

DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()


def test_logout():
    token.persist("A", "B", DATETIME_MAX)
    uri.persist("https://my-custom-environment")

    assert token.Filepath().exists()
    assert token.access(refresh=False) == "A"
    assert uri.Filepath().exists()
    assert uri.URI() == "https://my-custom-environment"

    logout()

    assert not token.Filepath().exists()
    assert not uri.Filepath().exists()
