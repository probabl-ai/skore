"""Skore-hub storage."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any
from uuid import UUID

import httpx

from .abstract_storage import AbstractStorage


class DirectoryDoesNotExist(Exception):
    """Directory does not exist."""


class SkoreHubStorage(AbstractStorage):
    def __init__(self, *, project_id: UUID, domain="http://0.0.0.0:8000"):
        self.url = f"{domain}/skore/projects"
        self.domain = domain
        self.project_id = project_id
        self.headers = {
            "Cookie": f"_oauth2_proxy={os.environ.get("SKORE_HUB_OAUTH_TOKEN", "")}"
        }

        print("YEAH")

    @classmethod
    def from_project_name(
        cls, name: str, *, domain="http://0.0.0.0:8000"
    ) -> SkoreHubStorage:
        url = f"{domain}/skore/projects"
        headers = {
            "Cookie": f"_oauth2_proxy={os.environ.get("SKORE_HUB_OAUTH_TOKEN", "")}"
        }
        response = httpx.post(f"{url}?name={name}", headers=headers)
        response.raise_for_status()
        project_id = response.json()["id"]

        return SkoreHubStorage(project_id=project_id, domain=domain)

    def __getitem__(self, key: str) -> Any:
        request = f"{self.url}/{self.project_id}/items/{key}/history"
        response = httpx.get(request, headers=self.headers)
        response.raise_for_status()

        response_json = response.json()

        return [
            {
                "item_class_name": row["item_class_name"],
                "item": row["item"],
            }
            for row in response_json
        ]

    def __setitem__(self, key: str, value: dict):
        request = f"{self.url}/{self.project_id}/items"
        data = {"key": key, **value}

        response = httpx.post(request, json=data, headers=self.headers)
        response.raise_for_status()

    def __delitem__(self, key: str):
        raise NotImplementedError

    def keys(self) -> Iterator[str]:
        request = f"{self.url}/{self.project_id}/keys"
        response = httpx.get(request, headers=self.headers)
        response.raise_for_status()

        yield from response.json()

    def values(self) -> Iterator[Any]:
        raise NotImplementedError

    def items(self) -> Iterator[tuple[str, Any]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
