import typing

import altair
import pydantic


class Vega(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, arbitrary_types_allowed=True)

    type: typing.Literal["vega"] = "vega"
    data: altair.vegalite.v5.api.Chart

    @pydantic.field_serializer("data")
    def serialize_data(self, data: altair.vegalite.v5.api.Chart) -> dict:
        return data.to_dict()
