"""Hub URL recording, independent of command wiring."""

from __future__ import annotations

import os

from skore._cli import skore as _skore
from skore._plugins.hub.authentication import HOST_ENV_VAR_NAME


def test_apply_hub_uri_records_a_resolved_url_without_discovery(monkeypatch):
    monkeypatch.delenv(HOST_ENV_VAR_NAME, raising=False)
    monkeypatch.setattr(
        _skore,
        "_discover_api_url",
        lambda url: (_ for _ in ()).throw(AssertionError(url)),
    )

    recorded = _skore.apply_hub_uri("http://hub.test")

    assert recorded == "http://hub.test"
    assert os.environ[HOST_ENV_VAR_NAME] == "http://hub.test"


def test_apply_hub_uri_prefers_a_discovered_api_url(monkeypatch):
    monkeypatch.setattr(_skore, "_discover_api_url", lambda url: "http://api.test")

    recorded = _skore.apply_hub_uri("http://frontend.test", discover=True)

    assert recorded == "http://api.test"
    assert os.environ[HOST_ENV_VAR_NAME] == "http://api.test"


def test_resolve_hub_uri_discovers_before_reading_the_canonical_uri(monkeypatch):
    monkeypatch.setattr(_skore, "_discover_api_url", lambda url: "http://api.test")

    assert _skore.resolve_hub_uri("http://frontend.test") == "http://api.test"
