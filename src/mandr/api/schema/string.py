import typing

import pydantic


class String(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["string"] = "string"
    data: str
