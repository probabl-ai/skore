from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot.metrics.feature_importance_display import (
    FeatureImportanceDisplay,
)
from skore.externals._pandas_accessors import DirNamesMixin

DataSource = Literal["test", "train", "X_y"]

Metric = Literal[
    "accuracy",
    "precision",
    "recall",
    "brier_score",
    "roc_auc",
    "log_loss",
    "r2",
    "rmse",
]


class _FeatureImportanceAccessor(_BaseAccessor[CrossValidationReport], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

    def coefficients(self) -> FeatureImportanceDisplay:
        coefficient_tables = []
        for split, report in enumerate(self._parent.estimator_reports_):
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

            estimator = (
                report_estimator[-1]
                if isinstance(report_estimator, Pipeline)
                else report_estimator
            )
            try:
                intercept = np.atleast_2d(estimator.intercept_)
            except AttributeError:
                intercept = np.atleast_2d(estimator.regressor_.intercept_)

            try:
                coef = np.atleast_2d(estimator.coef_)
            except AttributeError:
                coef = np.atleast_2d(estimator.regressor_.coef_)

            if intercept is None:
                data = coef.T
                index = list(feature_names)
            else:
                data = np.concatenate([intercept, coef.T])
                index = ["Intercept"] + list(feature_names)

            if data.shape[1] == 1:
                columns = [f"Coefficient_split_{split}"]
            elif is_classifier(report_estimator):
                columns = [f"Class #{i}" for i in range(data.shape[1])]
            else:
                columns = [f"Target #{i}" for i in range(data.shape[1])]

            df = pd.DataFrame(
                data=data,
                index=index,
                columns=columns,
            )
            coefficient_tables.append(df)

        combined = pd.concat(coefficient_tables, axis=1)

        return FeatureImportanceDisplay(combined, self._parent)

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
