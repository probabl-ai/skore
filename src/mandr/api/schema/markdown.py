import typing

import pydantic


class Markdown(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["markdown"] = "markdown"
    data: str
