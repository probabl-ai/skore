"""Hub URL recording, independent of command wiring."""

from __future__ import annotations

import os

from skore._cli import skore as _skore


def test_apply_hub_uri_records_a_resolved_url_without_discovery(monkeypatch):
    monkeypatch.delenv(_skore.URI_ENV, raising=False)
    monkeypatch.setattr(
        _skore,
        "_discover_api_url",
        lambda url: (_ for _ in ()).throw(AssertionError(url)),
    )

    recorded = _skore.apply_hub_uri("http://hub.test")

    assert recorded == "http://hub.test"
    assert os.environ[_skore.URI_ENV] == "http://hub.test"


def test_apply_hub_uri_prefers_a_discovered_api_url(monkeypatch):
    monkeypatch.setattr(_skore, "_discover_api_url", lambda url: "http://api.test")

    recorded = _skore.apply_hub_uri("http://frontend.test", discover=True)

    assert recorded == "http://api.test"
    assert os.environ[_skore.URI_ENV] == "http://api.test"


def test_resolve_hub_uri_discovers_before_reading_the_canonical_uri(monkeypatch):
    monkeypatch.setattr(_skore, "_discover_api_url", lambda url: "http://api.test")

    assert _skore.resolve_hub_uri("http://frontend.test") == "http://api.test"
