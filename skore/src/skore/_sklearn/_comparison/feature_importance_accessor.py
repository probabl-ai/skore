from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._plot.feature_importance.coefficients import CoefficientsDisplay
from skore._utils._accessor import _check_comparison_report_sub_estimators_have_coef

if TYPE_CHECKING:
    from skore import ComparisonReport


class _FeatureImportanceAccessor(_BaseAccessor["ComparisonReport"], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: ComparisonReport) -> None:
        super().__init__(parent)

    @available_if(_check_comparison_report_sub_estimators_have_coef())
    def coefficients(self) -> CoefficientsDisplay:
        """Retrieve the coefficients for each report, including the intercepts.

        If the compared reports are :class:`EstimatorReport` instances, the coefficients
        from each report's estimator are returned as a single-column DataFrame.

        If the compared reports are :class:`CrossValidationReport` instances, the
        coefficients across all cross-validation splits are retained and the columns are
        prefixed with the corresponding estimator name to distinguish them.

        Comparison reports with the same features are put under one key and are plotted
        together. When some reports share the same features and others do not, those
        with the same features are plotted together.

        Returns
        -------
        :class:`CoefficientsDisplay`
            The feature importance display containing model coefficients and
            intercept.
        """
        if self._parent._reports_type == "EstimatorReport":
            return CoefficientsDisplay._compute_data_for_display(
                estimators=[
                    report.estimator_ for report in self._parent.reports_.values()
                ],
                names=[name for name in self._parent.reports_.keys()],
                splits=[np.nan] * len(self._parent.reports_),
                report_type="comparison-estimator",
            )
        else:
            return CoefficientsDisplay._compute_data_for_display(
                estimators=[
                    estimator_report.estimator_
                    for cross_validation_report in self._parent.reports_.values()
                    for estimator_report in cross_validation_report.estimator_reports_
                ],
                names=list(
                    chain.from_iterable(
                        [
                            [name] * len(report.estimator_reports_)
                            for name, report in self._parent.reports_.items()
                        ]
                    )
                ),
                splits=list(
                    chain.from_iterable(
                        [i] * len(report.estimator_reports_)
                        for i, report in enumerate(self._parent.reports_.values())
                    )
                ),
                report_type="comparison-cross-validation",
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
        return self._rich_repr(class_name="skore.EstimatorReport.feature_importance")
