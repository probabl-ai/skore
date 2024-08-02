import typing

import pydantic


class Boolean(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["boolean"] = "boolean"
    data: bool
