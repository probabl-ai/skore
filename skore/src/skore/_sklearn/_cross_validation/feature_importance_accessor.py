from __future__ import annotations

import pandas as pd
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot.metrics.feature_importance_coefficients_display import (
    FeatureImportanceCoefficientsDisplay,
)
from skore._utils._accessor import _check_cross_validation_sub_estimator_has_coef


class _FeatureImportanceAccessor(_BaseAccessor[CrossValidationReport], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

    @available_if(_check_cross_validation_sub_estimator_has_coef())
    def coefficients(self) -> FeatureImportanceCoefficientsDisplay:
        """Retrieve the coefficients across splits, including the intercept.

        Returns
        -------
        :class:`FeatureImportanceCoefficientsDisplay`
            The feature importance display containing model coefficients and
            intercept.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = make_regression(n_features=3, random_state=42)
        >>> report = CrossValidationReport(
        >>>     estimator=Ridge(), X=X, y=y, splitter=5, n_jobs=4
        >>> )
        >>> display = report.feature_importance.coefficients()
        >>> display.frame()
                    Intercept	Feature #0	Feature #1	Feature #2
        Split index
        0       	0.064837	74.100966	27.309656	17.367865
        1       	0.030257	74.276481	27.571421	17.392395
        2       	0.000084	74.107126	27.614821	17.277730
        3       	0.145613	74.207645	27.523667	17.391055
        4       	0.033695	74.259575	27.599610	17.390481
        >>> display.plot() # shows plot
        """
        combined = pd.concat(
            {
                split: df["Coefficient"]
                for split, df in enumerate(
                    report.feature_importance.coefficients().frame()
                    for report in self._parent.estimator_reports_
                )
            },
            axis=1,
        ).T
        combined.index.name = "Split index"

        return FeatureImportanceCoefficientsDisplay("cross-validation", combined)

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _format_method_name(self, name: str) -> str:
        return f"{name}(...)".ljust(29)

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Available feature importance methods[/bold cyan]"

    def _get_help_tree_title(self) -> str:
        return "[bold cyan]report.feature_importance[/bold cyan]"

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(
            class_name="skore.CrossValidationReport.feature_importance"
        )
