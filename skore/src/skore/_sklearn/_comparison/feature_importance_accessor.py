from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.pipeline import Pipeline

from skore._sklearn._base import _BaseAccessor
from skore._sklearn._plot.metrics.feature_importance_display import (
    FeatureImportanceDisplay,
)
from skore.externals._pandas_accessors import DirNamesMixin

if TYPE_CHECKING:
    from skore._sklearn._comparison import ComparisonReport


class _FeatureImportanceAccessor(_BaseAccessor["ComparisonReport"], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: ComparisonReport) -> None:
        super().__init__(parent)

    def coefficients(self) -> FeatureImportanceDisplay:
        similar_reports = defaultdict(list)
        for report, name in zip(
            self._parent.reports_, self._parent.report_names_, strict=False
        ):
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

        coefficient_frames = []
        for reports_with_same_features in similar_reports.values():
            coef_dict = {}
            for report_data in reports_with_same_features:
                coefficients = (
                    report_data["report_obj"]
                    .feature_importance.coefficients()
                    .frame()
                    .iloc[:, 0]
                )
                coef_dict[report_data["estimator_name"]] = coefficients
            coef_frame = pd.DataFrame(coef_dict, index=report_data["feature_names"])
            coefficient_frames.append(coef_frame)

        return FeatureImportanceDisplay(coefficient_frames, self._parent)

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
