"""Schema to transfer listable value from store to dashboard."""

import typing

import pydantic


class Array(pydantic.BaseModel):
    """Schema to transfer listable value from store to dashboard.

    Examples
    --------
    >>> Array(data=[1, 2, 3])
    Array(...)

    >>> Array(data=(1, 2, 3))
    Array(...)

    >>> Array(type="array", data=(1, 2, 3))
    Array(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["array"] = "array"
    data: typing.Iterable
    metadata: typing.Optional[typing.Any] = None

    @pydantic.field_serializer("data")
    def serialize_data(self, data: typing.Iterable) -> list:
        """Serialize data from iterable to list."""
        return list(data)
