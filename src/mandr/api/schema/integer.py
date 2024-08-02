import typing

import pydantic


class Integer(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["integer"] = "integer"
    data: int
