from __future__ import annotations

from functools import cached_property
from typing import Any, Optional

from .. import item as item_module
from ..client.client import AuthenticatedClient


class Project:
    def __init__(self, tenant: str, name: str):
        self.__tenant = tenant
        self.__name = name

    @property
    def tenant(self):
        return self.__tenant

    @property
    def name(self):
        return self.__name

    @cached_property
    def run_id(self) -> str:
        with AuthenticatedClient(raises=True) as client:
            request = client.post(f"projects/{self.tenant}/{self.name}/runs")
            run = request.json()

            return run["id"]

    def put(
        self,
        key: str,
        value: Any,
        *,
        note: Optional[str] = None,
    ):
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string (found '{type(key)}')")

        if not isinstance(note, (type(None), str)):
            raise TypeError(f"Note must be a string (found '{type(note)}')")

        item = item_module.object_to_item(value)

        with AuthenticatedClient(raises=True) as client:
            client.post(
                f"projects/{self.tenant}/{self.name}/items/",
                json={
                    **item.__metadata__,
                    **item.__representation__,
                    **item.__parameters__,
                    "key": key,
                    "run_id": self.run_id,
                    "note": note,
                },
            )
