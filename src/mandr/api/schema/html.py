"""Schema to transfer HTML-like string value from store to dashboard."""

import typing

import pydantic


class HTML(pydantic.BaseModel):
    """Schema to transfer HTML-like string value from store to dashboard.

    Examples
    --------
    >>> HTML(data="<div></div>")
    HTML(...)

    >>> HTML(type="html", data="<div></div>")
    HTML(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["html"] = "html"
    data: str
    metadata: typing.Optional[typing.Any]
