from __future__ import annotations

from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._sklearn._plot.inspection.impurity_decrease import ImpurityDecreaseDisplay
from skore._utils._accessor import (
    _check_cross_validation_sub_estimator_has_coef,
    _check_cross_validation_sub_estimator_has_feature_importances,
)


class _InspectionAccessor(_BaseAccessor[CrossValidationReport], DirNamesMixin):
    """Accessor for model inspection related operations.

    You can access this accessor using the `inspection` attribute.
    """

    _verbose_name: str = "feature_importance"

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
        >>> report = CrossValidationReport(estimator=Ridge(), X=X, y=y, splitter=2)
        >>> display = report.inspection.coefficients()
        >>> display.frame()
            split     feature   coefficient
        0       0   Intercept       -0.1...
        1       0  Feature #0       73.2...
        2       0  Feature #1       26.6...
        3       0  Feature #2       17.1...
        4       1   Intercept        0.2...
        5       1  Feature #0       73.8...
        6       1  Feature #1       27.4...
        7       1  Feature #2       17.1...
        >>> display.plot() # shows plot
        """
        return CoefficientsDisplay._compute_data_for_display(
            estimators=[report.estimator_ for report in self._parent.reports_],
            names=[report.estimator_name_ for report in self._parent.reports_],
            splits=list(range(len(self._parent.reports_))),
            report_type="cross-validation",
        )

    @available_if(_check_cross_validation_sub_estimator_has_feature_importances())
    def impurity_decrease(self) -> ImpurityDecreaseDisplay:
        """Retrieve the Mean Decrease in Impurity (MDI) across splits.

        Returns
        -------
        :class:`ImpurityDecreaseDisplay`
            The impurity decrease display containing the feature importances.

        Examples
        --------
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from skore import CrossValidationReport
        >>> iris = load_iris(as_frame=True)
        >>> X, y = iris.data, iris.target
        >>> y = iris.target_names[y]
        >>> report = CrossValidationReport(
        ...     estimator=RandomForestClassifier(random_state=0), X=X, y=y, splitter=5
        ... )
        >>> display = report.inspection.impurity_decrease()
        >>> display.frame()
            split            feature  importance
        0       0  sepal length (cm)       0.0...
        1       0   sepal width (cm)       0.0...
        2       0  petal length (cm)       0.4...
        3       0   petal width (cm)       0.4...
        4       1  sepal length (cm)       0.0...
        ...
        >>> display.plot() # shows plot
        """
        return ImpurityDecreaseDisplay._compute_data_for_display(
            estimators=[report.estimator_ for report in self._parent.reports_],
            names=[report.estimator_name_ for report in self._parent.reports_],
            splits=list(range(len(self._parent.reports_))),
            report_type="cross-validation",
        )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.CrossValidationReport.inspection")
