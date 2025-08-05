from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._plot.metrics.feature_importance_display import (
    FeatureImportanceDisplay,
)
from skore._utils._accessor import _check_report_estimators_have_coef

if TYPE_CHECKING:
    from skore import ComparisonReport


class _FeatureImportanceAccessor(_BaseAccessor["ComparisonReport"], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: ComparisonReport) -> None:
        super().__init__(parent)

    @available_if(_check_report_estimators_have_coef())
    def coefficients(self) -> FeatureImportanceDisplay:
        """Retrieve the coefficients for each report, including the intercepts.

        If the input is a list of `EstimatorReport` objects, each estimator's
        coefficients are returned as a single-column DataFrame.

        If the input is a list of `CrossValidationReport` objects, the coefficients
        across all cross-validation splits are retained and the columns are prefixed
        with the corresponding estimator name to distinguish them.
        """
        similar_reports = defaultdict(list)
        from skore import CrossValidationReport, EstimatorReport

        for report, name in zip(
            self._parent.reports_, self._parent.report_names_, strict=False
        ):
            if isinstance(self._parent.reports_[0], CrossValidationReport):
                report = cast(CrossValidationReport, report)
                report_estimator = report.estimator_reports_[0].estimator_
            elif isinstance(self._parent.reports_[0], EstimatorReport):
                report_estimator = report.estimator_

            if isinstance(report_estimator, Pipeline):
                feature_names = report_estimator[:-1].get_feature_names_out()
            else:
                if hasattr(report_estimator, "feature_names_in_"):
                    feature_names = report_estimator.feature_names_in_
                else:
                    feature_names = [
                        f"Feature #{i}" for i in range(report_estimator.n_features_in_)
                    ]

            report_key = tuple(sorted(feature_names))
            similar_reports[report_key].append(
                {
                    "report_obj": report,
                    "estimator_name": name,
                    "feature_names": feature_names,
                }
            )

        coef_frames = []
        if isinstance(self._parent.reports_[0], EstimatorReport):
            for reports_with_same_features in similar_reports.values():
                coef_dict = {}
                for report_data in reports_with_same_features:
                    coef_dict[report_data["estimator_name"]] = (
                        report_data["report_obj"]
                        .feature_importance.coefficients()
                        .frame()
                        .iloc[:, 0]
                    )
                coef_frame = pd.DataFrame(coef_dict, index=report_data["feature_names"])
                coef_frames.append(coef_frame)

        elif isinstance(self._parent.reports_[0], CrossValidationReport):
            for reports_with_same_features in similar_reports.values():
                for report_data in reports_with_same_features:
                    coef_frames.append(
                        report_data["report_obj"]
                        .feature_importance.coefficients()
                        .frame()
                        .add_prefix(f"{report_data['estimator_name']}__")
                    )
        else:
            raise TypeError(f"Unexpected report type: {type(self._parent.reports_[0])}")

        return FeatureImportanceDisplay(coef_frames, self._parent)

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
