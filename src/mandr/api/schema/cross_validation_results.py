"""Schema to transfer sklearn-like cross-validation results from store to dashboard."""

from typing import Literal

# Some packages are necessary for the doctests
import altair  # noqa: F401
import pandas  # noqa: F401
from pydantic import BaseModel

from mandr.api.schema.dataframe import DataFrame
from mandr.api.schema.vega import Vega
from mandr.item.display_type import DisplayType


class CrossValidationResults(BaseModel):
    """Schema to transfer cross-validation results from store to dashboard.

    Examples
    --------
    >>> CrossValidationResults(
    ...     type="cv_results",
    ...     data=dict(
    ...         cv_results_table=DataFrame(data=pandas.DataFrame()),
    ...         test_score_plot=Vega(data=altair.Chart())
    ...     )
    ... )
    CrossValidationResults(...)

    >>> CrossValidationResults(
    ...     data=dict(
    ...         cv_results_table=DataFrame(data=pandas.DataFrame()),
    ...         test_score_plot=Vega(data=altair.Chart())
    ...     )
    ... )
    CrossValidationResults(...)
    """

    # We need to use a verbose name to make the pydantic validation errors transparent
    class CrossValidationResultsData(BaseModel):
        """The data in a CrossValidationResults."""

        cv_results_table: DataFrame
        test_score_plot: Vega

    type: Literal[DisplayType.CROSS_VALIDATION_RESULTS] = (
        DisplayType.CROSS_VALIDATION_RESULTS
    )
    data: CrossValidationResultsData
