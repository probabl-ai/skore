from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import pandas as pd
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation import CrossValidationReport
from skore._sklearn._estimator import EstimatorReport
from skore._sklearn._plot.metrics.feature_importance_display import (
    FeatureImportanceDisplay,
)
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
    def coefficients(self) -> FeatureImportanceDisplay:
        """Retrieve the coefficients for each report, including the intercepts.

        If the compared reports are `EstimatorReport`s, the coefficients from each
        report's estimator are returned as a single-column DataFrame.

        If the compared reports are `CrossValidationReport`s, the coefficients
        across all cross-validation splits are retained and the columns are prefixed
        with the corresponding estimator name to distinguish them.

        Comparison reports with the same features are put under one key and are plotted
        together.
        When some reports share the same features and others do not, those with the same
        features are plotted together.
        """
        similar_reports = defaultdict(list)

        for report, name in zip(
            self._parent.reports_, self._parent.report_names_, strict=False
        ):
            report = cast(CrossValidationReport | EstimatorReport, report)
            feature_names = (
                report.feature_importance.coefficients().frame().index.tolist()
            )
            similar_reports[tuple(sorted(feature_names))].append(
                {
                    "report_obj": report,
                    "estimator_name": name,
                    "feature_names": feature_names,
                }
            )

        coef_frames = []
        if self._parent._reports_type == "EstimatorReport":
            for reports_with_same_features in similar_reports.values():
                coef_frames.append(
                    pd.DataFrame(
                        {
                            report_data["estimator_name"]: (
                                report_data["report_obj"]
                                .feature_importance.coefficients()
                                .frame()
                                .iloc[:, 0]
                            )
                            for report_data in reports_with_same_features
                        },
                        index=reports_with_same_features[-1]["feature_names"],
                    )
                )

        elif self._parent._reports_type == "CrossValidationReport":
            for reports_with_same_features in similar_reports.values():
                for report_data in reports_with_same_features:
                    coef_frames.append(
                        report_data["report_obj"]
                        .feature_importance.coefficients()
                        .frame()
                        .add_prefix(f"{report_data['estimator_name']}__")
                    )
        else:
            raise TypeError(f"Unexpected report type: {self._parent._reports_type}")

        return FeatureImportanceDisplay(self._parent, coef_frames)

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
