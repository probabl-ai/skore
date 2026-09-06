from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np
from scipy.stats import spearmanr

from skore._checks.base import Check
from skore._checks.utils import (
    CheckNotApplicable,
    cast_report,
    get_preprocessed_X,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckCorrelatedFeatures(Check):
    """Check for highly correlated input features (SKD008).

    Flags when one or more pairs of numeric features have a Spearman rank
    correlation above 0.9.
    """

    code = "SKD008"
    title = "Highly correlated input features"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd008-correlated-features"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect pairs of numeric features with Spearman correlation above 0.9.

        Returns
        -------
        str or None
            Check result ``explanation`` when highly correlated features are
            detected; ``None`` when the check passes. Raises
            :class:`CheckNotApplicable` when feature data is unavailable or
            fewer than two numeric features are present.
        """
        report = cast_report(report)
        X = get_preprocessed_X(report, data_source="train")

        X = nw.from_native(X).select(nw.selectors.numeric())
        if X.shape[1] < 2 or X.shape[1] > 1000:
            raise CheckNotApplicable(
                "Expected train data to have between 2 and 1000 features; "
                f"got {X.shape[1]}."
            )

        corr_statistic = spearmanr(X.to_numpy()).statistic
        if X.shape[1] == 2:
            # With exactly 2 features, spearmanr returns a scalar, not a matrix.
            n_pairs = int(float(np.abs(corr_statistic)) >= 0.9)
        else:
            corr = np.abs(corr_statistic)
            np.fill_diagonal(corr, 0)
            n_pairs = int(np.count_nonzero(corr >= 0.9) // 2)

        if n_pairs:
            return (
                f"{n_pairs} pair(s) of features have a Spearman correlation "
                "above 0.9. Highly correlated features can destabilize "
                "linear model coefficients and feature-importance estimates, "
                "and may cause collinearity-induced numerical issues."
                "Dropping redundant features may also improve model performance."
            )
        return None
