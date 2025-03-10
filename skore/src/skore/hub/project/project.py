from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, Optional, Union

from skore.hub import item as item_module
from skore.hub.client.client import AuthenticatedClient
from skore.hub.item import object_to_item


class Project:
    def __init__(self, name: str, tenant: int):
        self.__name = name
        self.__tenant = tenant

    @property
    def name(self):
        return self.__name

    @property
    def tenant(self):
        return self.__tenant

    @cached_property
    def id(self):
        with AuthenticatedClient(raises=True) as client:
            # Retrieve existing project if exists
            request = client.get("projects/", params={"tenant_id": self.tenant})
            projects = request.json()

            for project in projects:
                if project["name"] == self.name:
                    return project["id"]

            # Create new project if not exists
            request = client.post(
                "projects/", json={"name": self.name, "tenant_id": self.tenant}
            )

            return request.json()["project_id"]

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

        now = datetime.now(tz=timezone.utc).replace(tzinfo=None).isoformat()
        item = object_to_item(value)

        with AuthenticatedClient(raises=True) as client:
            client.post(
                f"projects/{self.id}/items/",
                json={
                    "key": key,
                    "created_at": now,
                    "updated_at": now,
                    "value_type": item.__class__.__name__,
                    "value": {
                        "note": note,
                        "parameters": item.__parameters__,
                        "representation": item.__representation__.__dict__,
                    },
                },
            )

    def get(
        self,
        key: str,
        *,
        version: Optional[Union[Literal[-1, "all"], int]] = -1,
        metadata: bool = False,
    ):
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
