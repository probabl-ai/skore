from __future__ import annotations

# from datetime import datetime, timezone
from functools import cached_property
from typing import Any

from skore.hub.client import AuthenticatedClient
from skore.persistence import item as skore_item_module
from skore.persistence.item import item_to_object, object_to_item

# class ItemRecordModel(Base):
#     """Represents a skore item."""

#     __tablename__ = "item_record"

#     id = mapped_column(Integer, primary_key=True, autoincrement=True)
#     project_id = mapped_column(Integer, ForeignKey(ProjectModel.id), nullable=False)
#     key = mapped_column(String, index=True, nullable=False)
#     value = mapped_column(JSON, nullable=False)
#     value_type = mapped_column(String, nullable=False)
#     created_at = mapped_column(DateTime, nullable=False)
#     inserted_at = mapped_column(DateTime, default=func.now(), nullable=False)
#     updated_at = mapped_column(DateTime, onupdate=func.now(), nullable=False)


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
                "projects",
                follow_redirects=True,
                json={
                    "name": self.name,
                    "tenant_id": self.tenant,
                },
            )

            return request.json()

    def put(self, key: str, value: Any):
        # now_str = datetime.now(timezone.utc).isoformat()
        item = object_to_item(value)

        with AuthenticatedClient(raises=True) as client:
            client.post(
                f"projects/{self.id}/items",
                json={
                    "key": key,
                    # "created_at": now_str,
                    # "updated_at": now_str,
                    "item_class_name": item.__class__.__name__,
                    "item": {
                        "note": None,
                        "parameters": item.__parameters__,
                        # "representation": {
                        #     "media_type": "application/vnd.skore.estimator.report.lib.v1+json",
                        #     "value": {
                        #         "accuracy": 1,  # int
                        #         "rmse": 2,  # int
                        #         "plot": "effdsfds",  # base64_str
                        #     },
                        # },
                    },
                },
            )

    def get(self, key: str):
        # Retrieve item content from persistence
        with AuthenticatedClient(raises=True) as client:
            request = client.get(f"skore/projects/{self.id}/items/{key}")
            response = request.json()

        # Reconstruct item
        item_class = getattr(skore_item_module, response["item_class_name"])
        item = item_class(**response["item"]["parameters"])

        # Reconstruct original value
        return item_to_object(item)
