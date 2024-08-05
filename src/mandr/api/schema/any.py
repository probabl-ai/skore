"""Schema to transfer any JSON serializable value from store to dashboard."""

import typing

import pydantic


class Any(pydantic.BaseModel):
    """Schema to transfer any JSON serializable value from store to dashboard.

    Examples
    --------
    >>> Any(data=None)
    Any(...)

    >>> Any(type="any", data=None)
    Any(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["any"] = "any"
    data: typing.Any
