from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.utils.metaestimators import available_if

from skore._externals._pandas_accessors import DirNamesMixin
from skore._sklearn._base import _BaseAccessor
from skore._sklearn._cross_validation.report import CrossValidationReport
from skore._sklearn._plot.inspection.coefficients import CoefficientsDisplay
from skore._sklearn._plot.inspection.permutation_importance import (
    PermutationImportanceDisplay,
)
from skore._sklearn.types import DataSource
from skore._utils._accessor import _check_cross_validation_sub_estimator_has_coef

Metric = str | Callable | list[str] | tuple[str] | dict[str, Callable] | None


class _InspectionAccessor(_BaseAccessor[CrossValidationReport], DirNamesMixin):
    """Accessor for model inspection related operations.

    You can access this accessor using the `inspection` attribute.
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
        >>> display = report.inspection.coefficients()
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

    def permutation_importance(
        self,
        *,
        data_source: DataSource = "test",
        X: ArrayLike | None = None,
        y: ArrayLike | None = None,
        at_step: int | str = 0,
        metric: Metric = None,
        n_repeats: int = 5,
        max_samples: float = 1.0,
        n_jobs: int | None = None,
        seed: int | None = None,
    ) -> PermutationImportanceDisplay:
        importances = []
        for report_idx, report in enumerate(self._parent.estimator_reports_):
            display = report.inspection.permutation_importance(
                data_source=data_source,
                X=X,
                y=y,
                at_step=at_step,
                metric=metric,
                n_repeats=n_repeats,
                max_samples=max_samples,
                n_jobs=n_jobs,
                seed=seed,
            )
            df = display.importances
            df["split"] = report_idx
            importances.append(df)

        importances = pd.concat(importances, axis="index")
        return PermutationImportanceDisplay(
            importances=importances, report_type="cross-validation"
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
        return self._rich_repr(class_name="skore.CrossValidationReport.inspection")
