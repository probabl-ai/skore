import typing

import pydantic


class HTML(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["html"] = "html"
    data: str
