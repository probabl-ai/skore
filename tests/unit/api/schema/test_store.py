import datetime
import json
import pathlib

import altair
import numpy as np
import pandas
import pytest
from mandr.api import schema
from pydantic_core import PydanticSerializationError, ValidationError


class TestStore:
    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"type": "any", "data": 1}, {"type": "any", "data": 1}),
            (
                {"type": "array", "data": [1, 2, 3]},
                {"type": "array", "data": [1, 2, 3]},
            ),
            (
                {"type": "boolean", "data": False},
                {"type": "boolean", "data": False},
            ),
            (
                {
                    "type": "dataframe",
                    "data": pandas.DataFrame(
                        {
                            "column_string": ["a", "b", "c"],
                            "column_int": [1, 2, 3],
                            "column_float": [1.1, 2.2, 3.3],
                        },
                    ),
                },
                {
                    "type": "dataframe",
                    "data": {
                        "columns": ["column_string", "column_int", "column_float"],
                        "index": [0, 1, 2],
                        "data": [["a", 1, 1.1], ["b", 2, 2.2], ["c", 3, 3.3]],
                    },
                },
            ),
            (
                {"type": "date", "data": datetime.date(2024, 7, 31)},
                {"type": "date", "data": "2024-07-31"},
            ),
            (
                {"type": "datetime", "data": datetime.datetime(2024, 7, 31)},
                {"type": "datetime", "data": "2024-07-31T00:00:00"},
            ),
            (
                {"type": "file", "data": pathlib.Path("/my/file")},
                {"type": "file", "data": "/my/file"},
            ),
            (
                {"type": "html", "data": "hello"},
                {"type": "html", "data": "hello"},
            ),
            (
                {"type": "integer", "data": 1},
                {"type": "integer", "data": 1},
            ),
            (
                {"type": "markdown", "data": "hello"},
                {"type": "markdown", "data": "hello"},
            ),
            (
                {"type": "number", "data": 1.1},
                {"type": "number", "data": 1.1},
            ),
            (
                {"type": "number", "data": 1},
                {"type": "number", "data": 1},
            ),
            (
                {"type": "string", "data": "hello"},
                {"type": "string", "data": "hello"},
            ),
            (
                {
                    "type": "vega",
                    "data": altair.Chart(
                        pandas.DataFrame({"a": ["A"], "b": [28]})
                    ).mark_bar(),
                },
                {
                    "type": "vega",
                    "data": (
                        altair.Chart(pandas.DataFrame({"a": ["A"], "b": [28]}))
                        .mark_bar()
                        .to_dict()
                    ),
                },
            ),
            (
                {
                    "type": "vega",
                    "data": (
                        altair.Chart(
                            pandas.DataFrame({"a": ["A"], "b": [28]})
                        ).mark_bar()
                        + altair.Chart(
                            pandas.DataFrame({"a": ["A"], "b": [28]})
                        ).mark_bar()
                    ),
                },
                {
                    "type": "vega",
                    "data": (
                        altair.Chart(
                            pandas.DataFrame({"a": ["A"], "b": [28]})
                        ).mark_bar()
                        + altair.Chart(
                            pandas.DataFrame({"a": ["A"], "b": [28]})
                        ).mark_bar()
                    ).to_dict(),
                },
            ),
            (
                {"type": "numpy_array", "data": np.array([1, 2, 3, 4, 5])},
                {"type": "numpy_array", "data": np.array([1, 2, 3, 4, 5]).tolist()},
            ),
        ],
    )
    def test_payload(self, payload, expected):
        store = schema.Store(uri="/root", payload={"key": payload})
        dump = store.model_dump_json(by_alias=True)

        assert json.loads(dump) == {
            "schema": "schema:dashboard:v0",
            "uri": "/root",
            "payload": {
                "key": expected,
            },
        }

    @pytest.mark.parametrize(
        "payload,exception,match",
        [
            (
                {"type": "any", "data": pytest},
                PydanticSerializationError,
                "Unable to serialize",
            ),
            (
                {"type": "array", "data": None},
                ValidationError,
                "Input should be iterable",
            ),
            (
                {"type": "boolean", "data": None},
                ValidationError,
                "Input should be a valid boolean",
            ),
            (
                {"type": "dataframe", "data": None},
                ValidationError,
                "Input should be an instance of DataFrame",
            ),
            (
                {"type": "date", "data": datetime.datetime(2024, 7, 31)},
                ValidationError,
                "Input should be a valid date",
            ),
            (
                {"type": "datetime", "data": datetime.date(2024, 7, 31)},
                ValidationError,
                "Input should be a valid datetime",
            ),
            (
                {"type": "file", "data": None},
                ValidationError,
                "Input should be an instance of Path",
            ),
            (
                {"type": "html", "data": None},
                ValidationError,
                "Input should be a valid string",
            ),
            (
                {"type": "integer", "data": None},
                ValidationError,
                "Input should be a valid integer",
            ),
            (
                {"type": "markdown", "data": None},
                ValidationError,
                "Input should be a valid string",
            ),
            (
                {"type": "number", "data": None},
                ValidationError,
                "Input should be a valid number",
            ),
            (
                {"type": "string", "data": None},
                ValidationError,
                "Input should be a valid string",
            ),
            (
                {"type": "vega", "data": None},
                ValidationError,
                "Input should be an instance of Chart",
            ),
            (
                {"type": "numpy_array", "data": None},
                ValidationError,
                # quick fix to match pydantic_numpy error message
                "Input should be an instance of .*",
            ),
        ],
    )
    def test_payload_exception(self, payload, exception, match):
        with pytest.raises(exception, match=match):
            store = schema.Store(uri="/root", payload={"key": payload})
            store.model_dump_json(by_alias=True)
