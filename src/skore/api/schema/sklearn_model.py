"""Schema to transfer any JSON serializable value from store to dashboard."""

import typing

import pydantic
from sklearn.base import BaseEstimator, estimator_html_repr


class SKLearnModel(pydantic.BaseModel):
    """Schema to transfer a `sklearn.BaseEstimator` from store to dashboard.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> model = LinearRegression().fit(X, y)
    >>> SKLearnModel(data=model)
    SKLearnModel(...)
    """

    model_config = pydantic.ConfigDict(strict=True, arbitrary_types_allowed=True)

    type: typing.Literal["sklearn_model"] = "sklearn_model"
    data: BaseEstimator
    metadata: typing.Optional[typing.Any] = None

    @pydantic.field_serializer("data")
    def serialize_data(self, data: BaseEstimator):
        """Serialize a sklearn model to it's HTML representation."""
        return estimator_html_repr(data)
