from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

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


class CheckMDIHighCardinalityBias(Check):
    """Check for MDI bias with high-cardinality features (SKD007).

    Tips that mean-decrease-in-impurity importances may be inflated for
    continuous or high-cardinality features.

    We consider a feature to be high-cardinality when its number of unique values
    exceeds 50% of the number of samples.
    """

    code = "SKD007"
    title = "MDI biased for high-cardinality features"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd007-mdi-cardinality-bias"
    severity = "tip"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect high-cardinality features that may bias MDI importances."""
        report = cast_report(report)
        _, predictor = split_preprocessor_estimator(get_fitted_estimator(report))

        if not hasattr(predictor, "feature_importances_"):
            raise CheckNotApplicable(
                "Estimator is not a tree-based model: it does not have a "
                "`feature_importances_` attribute."
            )

        X = get_preprocessed_X(report, data_source="train")

        X = nw.from_native(X)
        n_samples = X.shape[0]
        high_cardinality_features = [
            column
            for column in X.columns
            if X.select(nw.col(column).n_unique()).item(0, 0) > 0.5 * n_samples
        ]

        if high_cardinality_features:
            names = ", ".join(str(s) for s in high_cardinality_features[:3])
            suffix = (
                f" (and {len(high_cardinality_features) - 3} more)"
                if len(high_cardinality_features) > 3
                else ""
            )
            return (
                f"High-cardinality features detected: {names}{suffix}. "
                "Mean Decrease in Impurity (MDI) importance is biased toward "
                "such features. Consider using permutation importance for "
                "a more robust alternative."
            )
        return None
