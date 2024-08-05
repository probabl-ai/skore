"""Schema to transfer filepath value from store to dashboard."""

import pathlib
import typing

import pydantic


class File(pydantic.BaseModel):
    """Schema to transfer filepath value from store to dashboard.

    Examples
    --------
    >>> File(data=pathlib.Path("/tmp/myfile.txt"))
    File(...)

    >>> File(type="file", data=pathlib.Path("/tmp/myfile.txt"))
    File(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["file"] = "file"
    data: pathlib.Path
