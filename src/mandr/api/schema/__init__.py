"""Schema to define endpoint requirements in the API."""

# Ignore import order because of CrossValidationResults
# ruff: noqa: I001

import typing

import pydantic
import typing_extensions

from mandr.api.schema.any import Any
from mandr.api.schema.array import Array
from mandr.api.schema.boolean import Boolean

from mandr.api.schema.dataframe import DataFrame
from mandr.api.schema.date import Date
from mandr.api.schema.datetime import Datetime
from mandr.api.schema.file import File
from mandr.api.schema.html import HTML
from mandr.api.schema.integer import Integer
from mandr.api.schema.markdown import Markdown
from mandr.api.schema.number import Number
from mandr.api.schema.string import String
from mandr.api.schema.vega import Vega

# Must be imported after DataFrame and Vega to prevent a circular import
from mandr.api.schema.cross_validation_results import CrossValidationResults

__all__ = [
    "Any",
    "Array",
    "Boolean",
    "CrossValidationResults",
    "DataFrame",
    "Date",
    "Datetime",
    "File",
    "HTML",
    "Integer",
    "Markdown",
    "Number",
    "Store",
    "String",
    "Vega",
]


class Store(pydantic.BaseModel):
    """Highest schema to transfer key-value pairs from store to dashboard.

    Examples
    --------
    >>> Store(uri="/root", payload={"key": {"type": "integer", "data": 0}})
    Store(...)
    """

    __NAME__ = "schema:dashboard:v0"

    model_config = pydantic.ConfigDict(strict=True)

    version: typing.Literal[__NAME__] = pydantic.Field(__NAME__, alias="schema")
    uri: str
    payload: dict[
        str,
        typing_extensions.Annotated[
            typing.Union[
                Any,
                Array,
                Boolean,
                CrossValidationResults,
                DataFrame,
                Date,
                Datetime,
                File,
                HTML,
                Integer,
                Markdown,
                Number,
                String,
                Vega,
            ],
            pydantic.Field(discriminator="type"),
        ],
    ]
