from __future__ import annotations

from functools import cached_property
from typing import Any

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

    def put(self, key: str, value: Any):
        item = object_to_item(value)

        with AuthenticatedClient(raises=True) as client:
            client.post(
                f"projects/{self.id}/items/",
                json={
                    "key": key,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at,
                    "value_type": item.__class__.__name__,
                    "value": {
                        "note": item.note,
                        "parameters": item.__parameters__,
                        "representation": item.__representation__.__dict__,
                    },
                },
            )

    def get(self, key: str):
        # Retrieve item content from persistence
        with AuthenticatedClient(raises=True) as client:
            request = client.get(f"projects/{self.id}/items/{key}")
            response = request.json()

        # Reconstruct item
        item_class = getattr(item_module, response["value_type"])
        item = item_class(**response["value"]["parameters"])

        # Reconstruct value
        return item.__raw__
