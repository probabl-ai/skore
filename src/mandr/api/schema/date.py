"""Schema to transfer date value from store to dashboard."""

import datetime
import typing

import pydantic


class Date(pydantic.BaseModel):
    """Schema to transfer date value from store to dashboard.

    Examples
    --------
    >>> Date(data=datetime.date(2024, 1, 1))
    Date(...)

    >>> Date(type="date", data=datetime.date(2024, 1, 1))
    Date(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["date"] = "date"
    data: datetime.date
    metadata: typing.Optional[typing.Any] = None
