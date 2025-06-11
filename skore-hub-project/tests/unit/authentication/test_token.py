import json
from datetime import datetime, timezone
from urllib.parse import urljoin

import pytest
from httpx import Response
from skore_hub_project.authentication.token import Token
from skore_hub_project.client.api import URI


DATETIME_MIN = datetime.min.replace(tzinfo=timezone.utc).isoformat()
DATETIME_MAX = datetime.max.replace(tzinfo=timezone.utc).isoformat()


class TestToken:
    @pytest.fixture
    def filepath(self, tmp_path):
        return tmp_path / "skore.token"

    def test_filepath(self, filepath):
        assert Token.filepath() == filepath

    def test_save(self, filepath):
        assert not filepath.exists()

        Token.save("A", "B", DATETIME_MAX)

        assert filepath.exists()
        assert json.loads(filepath.read_text()) == ["A", "B", DATETIME_MAX]

    def test_exists(self):
        assert not Token.exists()

        Token.save("A", "B", DATETIME_MAX)

        assert Token.exists()

    @pytest.mark.respx(assert_all_mocked=False)
    def test_access(self, respx_mock):
        assert not Token.exists()

        Token.save("A", "B", DATETIME_MAX)

        assert not respx_mock.calls
        assert Token.exists()
        assert Token.access() == "A"

    def test_access_expired(self, filepath, respx_mock, nowstr):
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

        assert not Token.exists()

        Token.save("A", "B", DATETIME_MIN)

        assert Token.exists()
        assert Token.access() == "D"
        assert json.loads(filepath.read_text()) == ["D", "E", DATETIME_MAX]
