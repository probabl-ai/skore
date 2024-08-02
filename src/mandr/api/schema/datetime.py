import datetime
import typing

import pydantic


class Datetime(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["datetime"] = "datetime"
    data: datetime.datetime
