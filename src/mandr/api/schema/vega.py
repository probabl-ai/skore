"""Schema to transfer (altair) VEGA-chart value from store to dashboard."""

import typing

import altair.vegalite.v5.schema.core
import pydantic


class Vega(pydantic.BaseModel):
    """Schema to transfer VEGA-chart value from store to dashboard.

    Examples
    --------
    >>> Vega(data=altair.Chart())
    Vega(...)

    >>> Vega(type="vega", data=altair.Chart())
    Vega(...)
    """

    model_config = pydantic.ConfigDict(strict=True, arbitrary_types_allowed=True)

    type: typing.Literal["vega"] = "vega"
    data: altair.vegalite.v5.schema.core.TopLevelSpec
    metadata: typing.Optional[typing.Any] = None

    @pydantic.field_serializer("data")
    def serialize_data(self, data: altair.vegalite.v5.api.Chart) -> dict:
        """Serialize data from `altair.vegalite.v5.api.Chart` to dict."""
        return data.to_dict()
