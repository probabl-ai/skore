"""Schema to transfer integer value from store to dashboard."""

import typing

import pydantic


class Integer(pydantic.BaseModel):
    """Schema to transfer integer value from store to dashboard.

    Examples
    --------
    >>> Integer(data=1)
    Integer(...)

    >>> Integer(type="integer", data=1)
    Integer(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["integer"] = "integer"
    data: int
    metadata: typing.Optional[typing.Any]
