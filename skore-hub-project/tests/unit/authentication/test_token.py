from datetime import datetime, timezone
from json import dumps, loads
from urllib.parse import urljoin

from httpx import Response
from pytest import fixture, mark, raises
from skore_hub_project.authentication import token
from skore_hub_project.authentication.uri import DEFAULT as URI

DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()


@fixture
def filepath(tmp_path):
    return tmp_path / "skore.token"


def test_post_oauth_refresh_token(respx_mock):
    route = respx_mock.post(urljoin(URI, "identity/oauth/token/refresh")).mock(
        Response(
            200,
            json={"access_token": "A", "refresh_token": "B", "expires_at": "C"},
        )
    )

    access_token, refresh_token, expires_at = token.post_oauth_refresh_token("token")

    assert route.calls.last.request.read() == b'{"refresh_token":"token"}'
    assert access_token == "A"
    assert refresh_token == "B"
    assert expires_at == "C"


def test_filepath(filepath):
    assert token.Filepath() == filepath


def test_persist(filepath):
    assert not filepath.exists()

    token.persist("A", "B", DATETIME_MAX)

    assert filepath.exists()
    assert loads(filepath.read_text()) == ["A", "B", DATETIME_MAX]


@mark.respx(assert_all_mocked=False)
def test_access(filepath, respx_mock):
    assert not filepath.exists()

    filepath.write_text(dumps(["A", "B", DATETIME_MAX]))

    assert token.access() == "A"
    assert filepath.exists()
    assert loads(filepath.read_text()) == ["A", "B", DATETIME_MAX]
    assert not respx_mock.calls


def test_access_expired(filepath, respx_mock):
    respx_mock.post(urljoin(URI, "identity/oauth/token/refresh")).mock(
        Response(
            200,
            json={
                "access_token": "D",
                "refresh_token": "E",
                "expires_at": DATETIME_MAX,
            },
        )
    )

    assert not filepath.exists()

    filepath.write_text(dumps(["A", "B", DATETIME_MIN]))

    assert filepath.exists()
    assert token.access() == "D"
    assert loads(filepath.read_text()) == ["D", "E", DATETIME_MAX]


@mark.respx(assert_all_mocked=False)
def test_access_exception(filepath, respx_mock):
    assert not filepath.exists()

    with raises(token.TokenError, match="not logged in"):
        token.access()

    assert not filepath.exists()
    assert not respx_mock.calls
