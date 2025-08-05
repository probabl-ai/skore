from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot.metrics.feature_importance_display import (
    FeatureImportanceDisplay,
)
from skore._utils._accessor import _check_estimator_report_has_coef


class _FeatureImportanceAccessor(_BaseAccessor[CrossValidationReport], DirNamesMixin):
    """Accessor for feature importance related operations.

    You can access this accessor using the `feature_importance` attribute.
    """

    def __init__(self, parent: CrossValidationReport) -> None:
        super().__init__(parent)

    @available_if(_check_estimator_report_has_coef())
    def coefficients(self) -> FeatureImportanceDisplay:
        """Retrieve the coefficients across splits, including the intercept.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from skore import CrossValidationReport
        >>> X, y = make_regression(n_features=3, random_state=42)
        >>> report = (
        >>>     estimator=Ridge(), X=X, y=y, cv_splitter=5, n_jobs=4
        >>> )
        >>> report.feature_importance.coefficients().frame()
                    Intercept	Feature #0	Feature #1	Feature #2
        Split index
        0       	-0.006675	63.354134	97.383292	56.331976
        1       	0.119352	63.356481	97.243977	56.258765
        2       	0.102091	63.119282	97.286351	56.124499
        3       	0.185287	63.032516	97.343662	56.398138
        4       	0.300477	62.997455	96.729395	56.075696
        >>> report.feature_importance.coefficients().plot() # shows plot
        """
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
            if hasattr(estimator, "intercept_"):
                intercept = np.atleast_2d(estimator.intercept_)
            else:
                intercept = np.atleast_2d(estimator.regressor_.intercept_)

            if hasattr(estimator, "coef_"):
                coef = np.atleast_2d(estimator.coef_)
            else:
                coef = np.atleast_2d(estimator.regressor_.coef_)

            if intercept is None:
                data = coef.T
                columns = list(feature_names)
            else:
                data = np.concatenate([intercept, coef.T])
                columns = ["Intercept"] + list(feature_names)

            if data.shape[1] == 1:
                index = [f"{split}"]
            elif is_classifier(report_estimator):
                index = [f"Class #{i}" for i in range(data.shape[1])]
            else:
                index = [f"Target #{i}" for i in range(data.shape[1])]

            coefficient_tables.append(
                pd.DataFrame(
                    data=data.T,
                    index=index,
                    columns=columns,
                )
            )

        combined = pd.concat(coefficient_tables)
        combined.index.name = "Split index"

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
