"""Schema to transfer string value from store to dashboard."""

import typing

import pydantic


class String(pydantic.BaseModel):
    """Schema to transfer string value from store to dashboard.

    Examples
    --------
    >>> String(data="value")
    String(...)

    >>> String(type="string", data="value")
    String(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["string"] = "string"
    data: str
