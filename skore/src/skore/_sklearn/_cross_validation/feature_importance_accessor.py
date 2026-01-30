from __future__ import annotations

from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot.feature_importance.coefficients import CoefficientsDisplay
from skore._utils._accessor import _check_cross_validation_sub_estimator_has_coef


class _FeatureImportanceAccessor(_BaseAccessor[CrossValidationReport], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

    @available_if(_check_cross_validation_sub_estimator_has_coef())
    def coefficients(self) -> CoefficientsDisplay:
        """Retrieve the coefficients across splits, including the intercept.

        Returns
        -------
        :class:`CoefficientsDisplay`
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
               feature  split  coefficients
        0   Feature #0      0       74.1...
        1   Feature #0      1       74.2...
        2   Feature #0      2       74.1...
        3   Feature #0      3       74.2...
        4   Feature #0      4       74.2...
        5   Feature #1      0       27.3...
        6   Feature #1      1       27.5...
        7   Feature #1      2       27.6...
        8   Feature #1      3       27.5...
        9   Feature #1      4       27.5...
        10  Feature #2      0       17.3...
        11  Feature #2      1       17.3...
        12  Feature #2      2       17.2...
        13  Feature #2      3       17.3...
        14  Feature #2      4       17.3...
        15   Intercept      0        0.0...
        16   Intercept      1        0.0...
        17   Intercept      2        0.0...
        18   Intercept      3        0.1...
        19   Intercept      4        0.0...
        >>> display.plot() # shows plot
        """
        return CoefficientsDisplay._compute_data_for_display(
            estimators=[
                report.estimator_ for report in self._parent.estimator_reports_
            ],
            names=[
                report.estimator_name_ for report in self._parent.estimator_reports_
            ],
            splits=list(range(len(self._parent.estimator_reports_))),
            report_type="cross-validation",
        )

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
