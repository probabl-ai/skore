import pathlib
import typing

import pydantic


class File(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["file"] = "file"
    data: pathlib.Path
