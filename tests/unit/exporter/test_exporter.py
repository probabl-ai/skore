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
        (1, {"display_type": DisplayType.ANY}, {"data": 1, "type": "any"}),
        (
            [1, 2, 3],
            {"display_type": DisplayType.ARRAY},
            {"data": [1, 2, 3], "type": "array"},
        ),
        (
            (1, 2, 3),
            {"display_type": DisplayType.ARRAY},
            {"data": [1, 2, 3], "type": "array"},
        ),
        (
            {1, 2, 3},
            {"display_type": DisplayType.ARRAY},
            {"data": [1, 2, 3], "type": "array"},
        ),
        (
            False,
            {"display_type": DisplayType.BOOLEAN},
            {"data": False, "type": "boolean"},
        ),
        (
            0,
            {"display_type": DisplayType.BOOLEAN},
            {"data": False, "type": "boolean"},
        ),
        (
            pandas.DataFrame(
                {
                    "column_string": ["a", "b", "c"],
                    "column_int": [1, 2, 3],
                    "column_float": [1.1, 2.2, 3.3],
                },
            ),
            {"display_type": DisplayType.DATAFRAME},
            {
                "data": {
                    "column_string": ["a", "b", "c"],
                    "column_int": [1, 2, 3],
                    "column_float": [1.1, 2.2, 3.3],
                },
                "type": "dataframe",
            },
        ),
        (
            datetime.datetime(2024, 7, 31),
            {"display_type": DisplayType.DATE},
            {"data": "2024-07-31", "type": "date"},
        ),
        # Test what happens when we pass a full datetime with DisplayType.DATE
        # (
        #     datetime.datetime(2024, 7, 31),
        #     {"display_type": DisplayType.DATE},
        #     {"data": "2024-07-31", "type": "date"},
        # ),
        (
            datetime.datetime(2024, 7, 31, 17, 43, 9, 435355),
            {"display_type": DisplayType.DATETIME},
            {"data": "2024-07-31T17:43:09.435355", "type": "datetime"},
        ),
        (
            pathlib.Path("/my/file"),
            {"display_type": DisplayType.FILE},
            {"data": "/my/file", "type": "file"},
        ),
        (
            "hello",
            {"display_type": DisplayType.HTML},
            {"data": "hello", "type": "html"},
        ),
        (
            1,
            {"display_type": DisplayType.INTEGER},
            {"data": 1, "type": "integer"},
        ),
        (
            "hello",
            {"display_type": DisplayType.MARKDOWN},
            {"data": "hello", "type": "markdown"},
        ),
        (
            1.1,
            {"display_type": DisplayType.NUMBER},
            {"data": 1.1, "type": "number"},
        ),
        (
            1,
            {"display_type": DisplayType.NUMBER},
            {"data": 1, "type": "number"},
        ),
        (
            "hello",
            {"display_type": DisplayType.STRING},
            {"data": "hello", "type": "string"},
        ),
        (
            altair.Chart(pandas.DataFrame({"a": ["A"], "b": [28]})).mark_bar(),
            {"display_type": DisplayType.VEGA},
            {
                "data": (
                    altair.Chart(pandas.DataFrame({"a": ["A"], "b": [28]}))
                    .mark_bar()
                    .to_dict()
                ),
                "type": "vega",
            },
        ),
    ],
)
def test_create_dto(value, metadata, expected):
    dto = exporter.create_item_dto(value, metadata)
    dump = dto.model_dump_json()

    assert json.loads(dump) == expected


@pytest.mark.parametrize(
    "value,metadata,exception,match",
    [
        (
            pytest,
            {"display_type": DisplayType.ANY},
            PydanticSerializationError,
            "Unable to serialize",
        ),
        (
            None,
            {"display_type": DisplayType.ARRAY},
            ValidationError,
            "Input should be iterable",
        ),
        (
            None,
            {"display_type": DisplayType.BOOLEAN},
            ValidationError,
            "Input should be a valid boolean",
        ),
        (
            None,
            {"display_type": DisplayType.DATAFRAME},
            ValidationError,
            "Input should be an instance of DataFrame",
        ),
        (
            None,
            {"display_type": DisplayType.DATE},
            ValidationError,
            "Input should be a valid date",
        ),
        (
            None,
            {"display_type": DisplayType.DATETIME},
            ValidationError,
            "Input should be a valid datetime",
        ),
        (
            None,
            {"display_type": DisplayType.FILE},
            ValidationError,
            "Input is not a valid path",
        ),
        (
            None,
            {"display_type": DisplayType.HTML},
            ValidationError,
            "Input should be a valid string",
        ),
        (
            None,
            {"display_type": DisplayType.INTEGER},
            ValidationError,
            "Input should be a valid integer",
        ),
        (
            None,
            {"display_type": DisplayType.MARKDOWN},
            ValidationError,
            "Input should be a valid string",
        ),
        (
            None,
            {"display_type": DisplayType.NUMBER},
            ValidationError,
            "Input should be a valid number",
        ),
        (
            None,
            {"display_type": DisplayType.STRING},
            ValidationError,
            "Input should be a valid string",
        ),
        (
            None,
            {"display_type": DisplayType.VEGA},
            ValidationError,
            "Input should be an instance of Chart",
        ),
    ],
)
def test_create_dto_exception(value, metadata, exception, match):
    with pytest.raises(exception, match=match):
        dto = exporter.create_item_dto(value, metadata)
        dump = dto.model_dump_json()


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
