"""Schema to transfer any JSON serializable value from store to dashboard."""

import base64
import typing
from io import StringIO

import matplotlib
import pydantic


class MatplotlibFigure(pydantic.BaseModel):
    """Schema to transfer a `matplotlib.figure.Figure` from store to dashboard.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()

    >>> MatplotlibFigure(data=fig)
    MatplotlibFigure(...)

    >>> MatplotlibFigure(type="matplotlib_figure", data=fig)
    MatplotlibFigure(...)
    """

    model_config = pydantic.ConfigDict(strict=True, arbitrary_types_allowed=True)

    type: typing.Literal["matplotlib_figure"] = "matplotlib_figure"
    data: matplotlib.figure.Figure

    @pydantic.field_serializer("data")
    def serialize_data(self, data: matplotlib.figure.Figure):
        """Serialize data from matplotlib Figure to SVG image."""
        output = StringIO()
        data.savefig(output, format="svg")
        image_string = output.getvalue()
        image_bytes = image_string.encode("utf-8")
        return base64.b64encode(image_bytes)
