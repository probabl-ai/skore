from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, Optional, Union

from .. import item as item_module
from ..client.client import AuthenticatedClient
from .metadata import Metadata


class Project:
    def __init__(self, tenant: str, name: str):
        self.__tenant = tenant
        self.__name = name

    @property
    def name(self):
        return self.__name

    @property
    def tenant(self):
        return self.__tenant

    @cached_property
    def run_id(self):
        with AuthenticatedClient(raises=True) as client:
            request = client.post(f"projects/{self.tenant}/{self.name}/runs")
            run = request.json()

            return run["id"]

    def metadata(self):
        return Metadata.factory(self)

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

    def get(
        self,
        key: str,
        *,
        version: Optional[Union[Literal[-1, "all"], int]] = -1,
        metadata: bool = False,
    ):
        raise NotImplementedError

        if not metadata:

            def dto(response):
                item_class = getattr(item_module, response["value_type"])
                item = item_class(**response["value"]["parameters"])
                return item.__raw__

        else:

            def dto(response):
                item_class = getattr(item_module, response["value_type"])
                item = item_class(**response["value"]["parameters"])
                return {
                    "value": item.__raw__,
                    "date": datetime.fromisoformat(response["created_at"]).replace(
                        tzinfo=timezone.utc
                    ),
                    "note": response["value"]["note"],
                }

        if version == -1:
            with AuthenticatedClient(raises=True) as client:
                request = client.get(f"projects/{self.id}/items/{key}")
                response = request.json()

            return dto(response)

        if version == "all" or (isinstance(version, int) and version >= 0):
            with AuthenticatedClient(raises=True) as client:
                request = client.get(f"projects/{self.id}/items/{key}/history")
                response = request.json()

            if version == "all":
                return list(map(dto, response))
            return dto(response[version])

        raise ValueError('`version` should be -1, "all", or a positive integer')
