from __future__ import annotations

from typing import TYPE_CHECKING, cast

import narwhals as nw

from skore._checks.base import Check
from skore._checks.utils import (
    CheckNotApplicable,
    cast_report,
    get_report_y,
)
from skore._utils.dataframe import UserSeries

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckHighClassImbalance(Check):
    """Check for high class imbalance (SKD004) in binary classification.

    Detects an issue when the most frequent class represents more than 80% of the
    dataset.
    """

    code = "SKD004"
    title = "High class imbalance"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd004-high-class-imbalance"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect when the majority class exceeds 80% of samples."""
        report = cast_report(report)
        if report.ml_task != "binary-classification":
            raise CheckNotApplicable(
                f"ML task is not binary classification. Got {report.ml_task}."
            )
        y = get_report_y(report, data_source="both")

        y = nw.from_native(cast(UserSeries, y), series_only=True)
        counts = y.value_counts()
        value_col = counts.columns[0]
        total = counts["count"].sum()
        overrepresented_class = counts.filter(nw.col("count") >= 0.8 * total)[
            value_col
        ].to_list()

        if len(overrepresented_class) > 0:
            return (
                f"Class {overrepresented_class} represents more than 80% of the "
                "dataset samples. Accuracy should not be used alone to assess model "
                "performance as it may be misleading by ignoring poor performance on "
                "the underrepresented class."
            )
        return None
