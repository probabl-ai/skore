"""Schema to transfer number value from store to dashboard."""

import typing

import pydantic


class Number(pydantic.BaseModel):
    """Schema to transfer number value from store to dashboard.

    Examples
    --------
    >>> Number(data=1.1)
    Number(...)

    >>> Number(type="number", data=1.1)
    Number(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["number"] = "number"
    data: float
    metadata: typing.Optional[typing.Any]
