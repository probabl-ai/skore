"""Credential failures reported by the hub helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from skore._cli.credentials import HubError, mint_api_key, require_workspace_api_key


def test_mint_api_key_reports_an_existing_key(monkeypatch):
    key_exists = type("KeyExistsError", (Exception,), {})

    def generate(**kwargs):
        raise key_exists()

    monkeypatch.setattr(
        "skore._cli.credentials._hub_key",
        lambda: SimpleNamespace(KeyExistsError=key_exists, generate=generate),
    )

    with pytest.raises(HubError, match="already stored"):
        mint_api_key(
            host=None,
            workspace="ws-1",
            name=None,
            expires="never",
            timeout=1,
            force=False,
        )


def test_mint_api_key_reports_permission_errors(monkeypatch):
    def generate(**kwargs):
        raise PermissionError("denied")

    key_exists = type("KeyExistsError", (Exception,), {})
    monkeypatch.setattr(
        "skore._cli.credentials._hub_key",
        lambda: SimpleNamespace(KeyExistsError=key_exists, generate=generate),
    )

    with pytest.raises(HubError, match="denied"):
        mint_api_key(
            host=None,
            workspace="ws-1",
            name=None,
            expires="never",
            timeout=1,
            force=False,
        )


def test_require_workspace_api_key_errors_when_mint_is_not_stored(monkeypatch):
    stored = {"key": None}

    def generate(**kwargs):
        stored["key"] = None

    monkeypatch.setattr(
        "skore._cli.credentials._hub_key",
        lambda: SimpleNamespace(
            KeyExistsError=type("KeyExistsError", (Exception,), {}),
            generate=generate,
            get=lambda **kwargs: stored["key"],
        ),
    )

    with pytest.raises(HubError, match="could not read an API key"):
        require_workspace_api_key(
            host="http://hub.test",
            workspace="ws-1",
            login_timeout=1,
        )
