from pytest import fixture


@fixture(autouse=True)
def setup_report(
    monkeypatch_hub_client,
    monkeypatch_project_routes,
    monkeypatch_upload_routes,
    monkeypatch_upload_with_mock,
): ...
