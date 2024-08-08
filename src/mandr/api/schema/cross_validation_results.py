"""Schema to transfer sklearn-like cross-validation results from store to dashboard."""

from typing import Literal

# need those for the doctests
import altair
import pandas
from pydantic import BaseModel

from mandr.api.schema import DataFrame, Vega
from mandr.item.display_type import DisplayType


class CrossValidationResults(BaseModel):
    """Schema to transfer cross-validation results from store to dashboard.

    Examples
    --------
    >>> CrossValidationResults(
        data=dict(
            cv_results_table=pandas.DataFrame(),
            test_score_plot=altair.Chart()
        )
    )
    CrossValidationResults(...)

    >>> CrossValidationResults(
        type="cv_results",
        data=dict(
            cv_results_table=pandas.DataFrame(),
            test_score_plot=altair.Chart()
        )
    )
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
