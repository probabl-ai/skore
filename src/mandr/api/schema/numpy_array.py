"""Schema to transfer listable value from store to dashboard."""

import typing

import numpy as np
import pydantic
import pydantic_numpy.typing as pnd


class NumpyArray(pydantic.BaseModel):
    """Schema to transfer NumPy array value from store to dashboard.

    Examples
    --------
    >>> NumpyArray(data=np.random.randint(0, 100, size=50))
    NumpyArray(...)

    >>> NumpyArray(data=np.random.randint(0, 100, size=50))
    NumpyArray(...)

    >>> NumpyArray(type="numpy_array", data=np.random.randint(0, 100, size=50))
    NumpyArray(...)
    """

    model_config = pydantic.ConfigDict(strict=True)

    type: typing.Literal["numpy_array"] = "numpy_array"
    data: pnd.NpNDArray
    metadata: typing.Optional[typing.Any]

    @pydantic.field_serializer("data")
    def serialize_data(self, data: np.ndarray) -> list:
        """Serialize data from ndarray to list."""
        return data.tolist()
