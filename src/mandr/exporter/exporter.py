from __future__ import annotations

import datetime
import pathlib
import typing
from typing import TYPE_CHECKING, Literal, Union

import altair
import pandas
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer
from typing_extensions import Annotated

from mandr.item.display_type import DisplayType

if TYPE_CHECKING:
    from mandr.store import Store


class Any(BaseModel):
    type: Literal[DisplayType.ANY] = DisplayType.ANY
    data: typing.Any


class Array(BaseModel):
    type: Literal[DisplayType.ARRAY] = DisplayType.ARRAY
    data: typing.Iterable

    @field_serializer("data")
    def serialize_data(self, data: typing.Iterable) -> list:
        return list(data)


class Boolean(BaseModel):
    type: Literal[DisplayType.BOOLEAN] = DisplayType.BOOLEAN
    data: bool


class DataFrame(BaseModel):
    type: Literal[DisplayType.DATAFRAME] = DisplayType.DATAFRAME
    data: pandas.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("data")
    def serialize_data(self, data: pandas.DataFrame) -> dict:
        return data.to_dict(orient="list")


class Date(BaseModel):
    type: Literal[DisplayType.DATE] = DisplayType.DATE
    data: datetime.date


class Datetime(BaseModel):
    type: Literal[DisplayType.DATETIME] = DisplayType.DATETIME
    data: datetime.datetime


class File(BaseModel):
    type: Literal[DisplayType.FILE] = DisplayType.FILE
    data: pathlib.Path


class HTML(BaseModel):
    type: Literal[DisplayType.HTML] = DisplayType.HTML
    data: str


class Integer(BaseModel):
    type: Literal[DisplayType.INTEGER] = DisplayType.INTEGER
    data: int


class Markdown(BaseModel):
    type: Literal[DisplayType.MARKDOWN] = DisplayType.MARKDOWN
    data: str


class Number(BaseModel):
    type: Literal[DisplayType.NUMBER] = DisplayType.NUMBER
    data: float


class String(BaseModel):
    type: Literal[DisplayType.STRING] = DisplayType.STRING
    data: str


class Vega(BaseModel):
    type: Literal[DisplayType.VEGA] = DisplayType.VEGA
    data: altair.vegalite.v5.api.Chart

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("data")
    def serialize_data(self, data: altair.vegalite.v5.api.Chart) -> dict:
        return data.to_dict()


ItemDto = Annotated[
    Union[
        Any,
        Array,
        Boolean,
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
    Field(discriminator="type"),
]


# def create_item_dto(value, type=None) -> ItemDto:
def create_item_dto(value: Any, metadata: dict) -> ItemDto:
    """Construct an ItemDto."""
    # TypeAdapter is the Pydantic v2 way to build objects made from Unions.
    # See <https://blog.det.life/pydantic-for-experts-discriminated-unions-in-pydantic-v2-2d9ca965b22f>
    ta = TypeAdapter(ItemDto)
    # obj = {"type": str(type or DisplayType.infer(value)), "data": value}
    # return ta.validate_python(obj)
    return ta.validate_python({"type": metadata["display_type"], "data": value})


class StoreDto(BaseModel):
    """A serialized form of the store."""

    schema_: Literal["schema:dashboard:v0"] = Field(
        "schema:dashboard:v0", alias="schema"
    )
    uri: str
    payload: dict[str, ItemDto]

    @staticmethod
    def from_store(store: Store):
        """Serialize all `Item`s in `store`."""
        return StoreDto(
            uri=str(store.uri),
            payload={
                key: create_item_dto(value, metadata)
                for key, value, metadata in store.items(metadata=True)
            },
        )
