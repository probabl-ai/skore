"""Schema to transfer datetime value from store to dashboard."""

import datetime
import typing

import pydantic


class Datetime(pydantic.BaseModel):
    """Schema to transfer datetime value from store to dashboard.

    Examples
    --------
    >>> Datetime(data=datetime.datetime(2024, 1, 1, 0, 0, 0))
    Datetime(...)

    >>> Datetime(type="datetime", data=datetime.datetime(2024, 1, 1, 0, 0, 0))
    Datetime(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["datetime"] = "datetime"
    data: datetime.datetime
    metadata: typing.Optional[typing.Any] = None
