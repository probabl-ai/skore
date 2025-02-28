from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from typing import Any

from skore.hub.client.client import AuthenticatedClient
from skore.hub.item import item as skore_item_module
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
            request = client.post(
                "projects/", json={"name": self.name, "tenant_id": self.tenant}
            )

            return request.json()

    def put(self, key: str, value: Any):
        now = datetime.now(timezone.utc).isoformat()
        item = object_to_item(value)

        with AuthenticatedClient(raises=True) as client:
            client.post(
                f"projects/{self.id}/items/",
                json={
                    "key": key,
                    "created_at": now,
                    "updated_at": now,
                    "value_type": None,
                    "value": {
                        "note": None,
                        "type": item.__class__.__name__,
                        "parameters": item.__parameters__,
                        "representation": item.__representation__.__dict__,
                    },
                },
            )

    def get(self, key: str):
        # Retrieve item content from persistence
        with AuthenticatedClient(raises=True) as client:
            request = client.get(f"skore/projects/{self.id}/items/{key}")
            response = request.json()

        # Reconstruct item
        item_class = getattr(skore_item_module, response["value"]["type"])
        item = item_class(**response["value"]["parameters"])

        # Reconstruct value
        return item.__raw__
