import typing

import pydantic


class Number(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["number"] = "number"
    data: float
