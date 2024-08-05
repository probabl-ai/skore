"""Schema to transfer `pandas.DataFrame` value from store to dashboard."""

import typing

import pandas
import pydantic


class DataFrame(pydantic.BaseModel):
    """Schema to transfer `pandas.DataFrame` value from store to dashboard.

    Examples
    --------
    >>> DataFrame(data=pandas.DataFrame())
    DataFrame(...)

    >>> DataFrame(type="dataframe", data=pandas.DataFrame())
    DataFrame(...)
    """

    model_config = pydantic.ConfigDict(strict=True, arbitrary_types_allowed=True)

    type: typing.Literal["dataframe"] = "dataframe"
    data: pandas.DataFrame

    @pydantic.field_serializer("data")
    def serialize_data(self, data: pandas.DataFrame) -> dict:
        """Serialize data from `pandas.DataFrame` to dict."""
        return data.to_dict(orient="list")
