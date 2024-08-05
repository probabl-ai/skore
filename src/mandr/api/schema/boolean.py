"""Schema to transfer boolean value from store to dashboard."""

import typing

import pydantic


class Boolean(pydantic.BaseModel):
    """Schema to transfer boolean value from store to dashboard.

    Examples
    --------
    >>> Boolean(data=True)
    Boolean(...)

    >>> Boolean(type="boolean", data=True)
    Boolean(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["boolean"] = "boolean"
    data: bool
