import typing

import pydantic


class Array(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["array"] = "array"
    data: typing.Iterable

    @pydantic.field_serializer("data")
    def serialize_data(self, data: typing.Iterable) -> list:
        return list(data)
