from fastapi.testclient import TestClient
from mandr.infomander import InfoMander


def test_index(client: TestClient):
    response = client.get("/")
    assert "html" in response.headers["Content-Type"]
    assert response.status_code == 200


def test_get_mandrs(client: TestClient):
    number_of_manders = 5
    for i in range(5):
        mander = InfoMander(f"probabl-ai/test-mandr/{i}")
        mander.add_info("hey", "ho")

    response = client.get("/api/mandrs")
    mander_paths = response.json()
    assert len(mander_paths) == number_of_manders
    assert response.status_code == 200


def test_get_mandr(client: TestClient):
    mander_path = "probabl-ai/test-mandr/42"
    mander = InfoMander(mander_path)
    mander.add_info("hey", "ho let's go")

    response = client.get(f"/api/mandrs/{mander_path}")
    mander_json = response.json()
    assert mander_path in mander_json.get("path")
    assert response.status_code == 200

    response = client.get("/api/mandrs/i/do/not/exists")
    assert response.status_code == 404
