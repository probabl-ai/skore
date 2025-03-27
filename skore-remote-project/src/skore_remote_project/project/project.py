from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from typing import Any, Literal, Optional, Union

from .. import item as item_module
from ..client.client import AuthenticatedClient


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


import pandas as pd


class Metadata(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self.index, pd.RangeIndex):
            self.set_index("report_id", inplace=True)

        self.index.name = None

    @property
    def _constructor(self):
        return Metadata

    def reports(self):
        return [f"REPORT {index}" for index in self.index]


if __name__ == "__main__":
    df = Metadata(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9],
            "report_id": ["RID-1", "RID-2", "RID-3"],
        }
    )
    df.iloc[1:2].reports()
