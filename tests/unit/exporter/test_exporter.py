import datetime
import json
import pathlib

import altair
import pandas
import pytest
from mandr.exporter import exporter
from mandr.item import DisplayType, Item, ItemMetadata
from mandr.storage import URI, NonPersistentStorage
from mandr.store import Store
from pydantic_core import PydanticSerializationError, ValidationError


@pytest.mark.parametrize(
    "value,metadata,expected",
    [
        (
            # NOTE: Not a valid cross-validation results dict, but here what matters is
            # only the "computed" metadata
            {"fit_time": [0.5, 0.2]},
            {
                "display_type": DisplayType.CROSS_VALIDATION_RESULTS,
                "computed": {
                    "cv_results_table": exporter.DataFrame(data=pandas.DataFrame()),
                    "test_score_plot": exporter.Vega(data=altair.Chart().mark_bar()),
                },
            },
            {
                "data": {
                    "cv_results_table": {
                        "data": {"columns": [], "data": [], "index": []},
                        "type": "dataframe",
                    },
                    "test_score_plot": {
                        "data": altair.Chart().mark_bar().to_dict(),
                        "type": "vega",
                    },
                },
                "type": "cv_results",
            },
        ),
    ],
)
def test_create_dto(value, metadata, expected):
    dto = exporter.create_item_dto(value, metadata)
    dump = dto.model_dump_json()

    assert json.loads(dump) == expected


def test_store_dto(mock_nowstr):
    storage = NonPersistentStorage(
        content={
            URI("/root/key"): Item(
                data="value",
                metadata=ItemMetadata(
                    display_type=DisplayType.STRING,
                    created_at=mock_nowstr,
                    updated_at=mock_nowstr,
                ),
            )
        }
    )

    store = Store("root", storage=storage)
    dto = exporter.StoreDto.from_store(store)
    dump = dto.model_dump_json(by_alias=True)

    assert json.loads(dump) == {
        "schema": "schema:dashboard:v0",
        "uri": "/root",
        "payload": {"key": {"type": "string", "data": "value"}},
    }


def test_store_dto_cross_validation_results(mock_nowstr):
    store = Store("root", storage=NonPersistentStorage())
    store.insert("my_cv", {"test_score": [1, 2]}, display_type="cv_results")

    dto = exporter.StoreDto.from_store(store)
    dump = dto.model_dump_json(by_alias=True)

    test_score_plot = store.read("my_cv", metadata=True)[1]["computed"][
        "test_score_plot"
    ]

    assert json.loads(dump) == {
        "schema": "schema:dashboard:v0",
        "uri": "/root",
        "payload": {
            "my_cv": {
                "type": "cv_results",
                "data": {
                    "cv_results_table": {
                        "type": "dataframe",
                        "data": {
                            "columns": ["test_score"],
                            "data": [[1], [2]],
                            "index": [0, 1],
                        },
                    },
                    "test_score_plot": test_score_plot.model_dump(),
                },
            }
        },
    }
