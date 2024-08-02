import typing

import pydantic


class Any(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["any"] = "any"
    data: typing.Any
