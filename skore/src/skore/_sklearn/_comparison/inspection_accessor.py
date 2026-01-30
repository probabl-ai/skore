from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._utils._accessor import _check_comparison_report_sub_estimators_have_coef

if TYPE_CHECKING:
    from skore import ComparisonReport
    from skore._sklearn._cross_validation.report import CrossValidationReport


class _InspectionAccessor(_BaseAccessor["ComparisonReport"], DirNamesMixin):
    """Accessor for model inspection related operations.

    You can access this accessor using the `inspection` attribute.
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

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.linear_model import Ridge
        >>> from skore import train_test_split
        >>> from skore import ComparisonReport, EstimatorReport
        >>> X, y = load_diabetes(return_X_y=True)
        >>> split_data = train_test_split(X=X, y=y, shuffle=False, as_dict=True)
        >>> report_big_alpha = EstimatorReport(Ridge(alpha=1e3), **split_data)
        >>> report_small_alpha = EstimatorReport(Ridge(alpha=1e-3), **split_data)
        >>> report = ComparisonReport({
        ...     "report small alpha": report_small_alpha,
        ...     "report big alpha": report_big_alpha,
        ... })
        >>> display = report.inspection.coefficients()
        >>> display.frame()
                    estimator     feature  coefficients
        0   report small alpha   Intercept    151.5...
        1   report small alpha  Feature #0    -11.6...
        2   report small alpha  Feature #1   -238.2...
        3   report small alpha  Feature #2    505.5...
        4   report small alpha  Feature #3    298.4...
        5   report small alpha  Feature #4   -408.5...
        6   report small alpha  Feature #5    164.0...
        7   report small alpha  Feature #6    -55.1...
        8   report small alpha  Feature #7    123.0...
        9   report small alpha  Feature #8    633.8...
        10  report small alpha  Feature #9     99.4...
        11    report big alpha   Intercept    151.0...
        12    report big alpha  Feature #0      0.2...
        13    report big alpha  Feature #1      0.0...
        14    report big alpha  Feature #2      0.6...
        15    report big alpha  Feature #3      0.5...
        16    report big alpha  Feature #4      0.2...
        17    report big alpha  Feature #5      0.2...
        18    report big alpha  Feature #6     -0.4...
        19    report big alpha  Feature #7      0.5...
        20    report big alpha  Feature #8      0.6...
        21    report big alpha  Feature #9      0.4...
        >>> display.plot() # shows plot
        """
        if self._parent._reports_type == "EstimatorReport":
            return CoefficientsDisplay._compute_data_for_display(
                estimators=[
                    report.estimator_ for report in self._parent.reports_.values()
                ],
                names=list(self._parent.reports_.keys()),
                splits=[np.nan] * len(self._parent.reports_),
                report_type="comparison-estimator",
            )
        else:  # self._parent._reports_type == "CrossValidationReport":
            estimators, names = [], []
            splits: list[int | float] = []
            for name, report in self._parent.reports_.items():
                cross_validation_report = cast("CrossValidationReport", report)
                for split_idx, estimator_report in enumerate(
                    cross_validation_report.estimator_reports_
                ):
                    estimators.append(estimator_report.estimator_)
                    names.append(name)
                    splits.append(split_idx)
            return CoefficientsDisplay._compute_data_for_display(
                estimators=estimators,
                names=names,
                splits=splits,
                report_type="comparison-cross-validation",
            )

    ####################################################################################
    # Methods related to the help tree
    ####################################################################################

    def _format_method_name(self, name: str) -> str:
        return f"{name}(...)".ljust(29)

    def _get_help_panel_title(self) -> str:
        return "[bold cyan]Available model inspection methods[/bold cyan]"

    def _get_help_tree_title(self) -> str:
        return "[bold cyan]report.inspection[/bold cyan]"

    def __repr__(self) -> str:
        """Return a string representation using rich."""
        return self._rich_repr(class_name="skore.EstimatorReport.inspection")
