"""In-memory storage."""

from collections.abc import Iterator
from typing import Any

import httpx

from .abstract_storage import AbstractStorage


class DirectoryDoesNotExist(Exception):
    """Directory does not exist."""


class SkoreHubStorage(AbstractStorage):
    def __init__(self, name: str, *, domain="0.0.0.0:8000"):
        self.url = f"http://{domain}/skore/projects"
        self.domain = domain

        response = httpx.post(f"{self.url}?name={name}")
        response.raise_for_status()

        self.id = response.json()["id"]

    def __getitem__(self, key: str) -> Any:
        request = f"{self.url}/{self.id}/items/{key}/history"
        response = httpx.get(request)
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
        request = f"{self.url}/{self.id}/items"
        data = {"key": key, **value}

        response = httpx.post(request, json=data)
        response.raise_for_status()

    def __delitem__(self, key: str):
        raise NotImplementedError

    def keys(self) -> Iterator[str]:
        request = f"{self.url}/{self.id}/keys"
        response = httpx.get(request)
        response.raise_for_status()

        yield from response.json()

    def values(self) -> Iterator[Any]:
        raise NotImplementedError

    def items(self) -> Iterator[tuple[str, Any]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
