from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np

from skore._checks.base import Check
from skore._checks.utils import (
    CheckNotApplicable,
    cast_report,
    get_fitted_estimator,
    get_preprocessed_X,
    split_preprocessor_estimator,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckCoefficientsInterpretation(Check):
    """Check coefficient interpretability for linear models (SKD006).

    Tips about whether coefficients can be compared across features and
    whether they retain their original-unit interpretation.
    """

    code = "SKD006"
    title = "Coefficient interpretation"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd006-unscaled-coefficients"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        """Assess whether linear-model coefficients are comparable and interpretable."""
        report = cast_report(report)
        _, predictor = split_preprocessor_estimator(get_fitted_estimator(report))

        if not hasattr(predictor, "coef_"):
            raise CheckNotApplicable(
                "Estimator is not a linear model: it does not have a `coef_` attribute."
            )

        X = get_preprocessed_X(report, data_source="both")

        std_values = nw.from_native(X).select(nw.all().std()).to_numpy().ravel()
        if not np.allclose(std_values, std_values[0], atol=0.05):
            return (
                "Features are not on the same scale: coefficient magnitudes "
                "are not directly comparable as feature importance."
            )
        return (
            "Features appear to be standardized: coefficients are comparable "
            "but no longer interpretable in the original feature units."
        )
