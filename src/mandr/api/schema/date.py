import datetime
import typing

import pydantic


class Date(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["date"] = "date"
    data: datetime.date
