import os

from fastapi.testclient import TestClient
from mandr.infomander import InfoMander


def test_index(client: TestClient):
    response = client.get("/")

    assert "html" in response.headers["Content-Type"]
    assert response.status_code == 200


def test_list_mandrs(client: TestClient, tmp_path):
    os.environ["MANDR_PATH"] = "root"
    os.environ["MANDR_ROOT"] = str(tmp_path)

    InfoMander("root", root=tmp_path).add_info("key", "value")
    InfoMander("root/subroot1", root=tmp_path).add_info("key", "value")
    InfoMander("root/subroot2", root=tmp_path).add_info("key", "value")
    InfoMander("root/subroot2/subsubroot1", root=tmp_path).add_info("key", "value")
    InfoMander("root/subroot2/subsubroot2", root=tmp_path).add_info("key", "value")

    response = client.get("/mandrs")

    assert response.status_code == 200
    assert response.json() == [
        "root",
        "root/subroot1",
        "root/subroot2",
        "root/subroot2/subsubroot1",
        "root/subroot2/subsubroot2",
    ]


def test_get_mandr(monkeypatch, mock_now, mock_nowstr, client: TestClient, tmp_path):
    class MockDatetime:
        @staticmethod
        def now(*args, **kwargs):
            return mock_now

    monkeypatch.setattr("mandr.infomander.datetime", MockDatetime)

    os.environ["MANDR_ROOT"] = str(tmp_path)

    InfoMander("root", root=tmp_path).add_info("key", None)
    InfoMander("root/subroot1", root=tmp_path).add_info("key", None)
    InfoMander("root/subroot2", root=tmp_path).add_info("key", None)
    InfoMander("root/subroot2/subsubroot1", root=tmp_path).add_info("key", None)
    InfoMander("root/subroot2/subsubroot2", root=tmp_path).add_info("key", "value")

    response = client.get("/mandrs/root/subroot2/subsubroot2")

    assert response.status_code == 200
    assert response.json() == {
        "path": "root/subroot2/subsubroot2",
        "views": {},
        "logs": {},
        "artifacts": {},
        "info": {
            "key": "value",
            "updated_at": mock_nowstr,
        },
    }

    response = client.get("/mandrs/root/subroot2/subroot3")

    assert response.status_code == 404
