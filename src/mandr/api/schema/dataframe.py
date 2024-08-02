import typing

import pandas
import pydantic


class DataFrame(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(strict=True, arbitrary_types_allowed=True)

    type: typing.Literal["dataframe"] = "dataframe"
    data: pandas.DataFrame

    @pydantic.field_serializer("data")
    def serialize_data(self, data: pandas.DataFrame) -> dict:
        return data.to_dict(orient="list")
