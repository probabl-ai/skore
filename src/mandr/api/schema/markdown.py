"""Schema to transfer Markdown-like string value from store to dashboard."""

import typing

import pydantic


class Markdown(pydantic.BaseModel):
    """Schema to transfer Markdown-like string value from store to dashboard.

    Examples
    --------
    >>> Markdown(data="# title")
    Markdown(...)

    >>> Markdown(type="markdown", data="# title")
    Markdown(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["markdown"] = "markdown"
    data: str
