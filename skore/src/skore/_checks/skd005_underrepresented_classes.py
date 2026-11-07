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


class CheckUnderrepresentedClasses(Check):
    """Check for underrepresented classes (SKD005) in multiclass classification.

    Detects an issue when some classes represent less than 10% of the dataset.
    """

    code = "SKD005"
    title = "Underrepresented classes"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd005-underrepresented-classes"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect classes that each represent less than 10% of samples."""
        report = cast_report(report)
        if report.ml_task != "multiclass-classification":
            raise CheckNotApplicable(
                f"ML task is not multiclass classification. Got {report.ml_task}."
            )

        y = get_report_y(report, data_source="both")

        y = nw.from_native(cast(UserSeries, y), series_only=True)
        counts = y.value_counts()
        value_col = counts.columns[0]
        total = counts["count"].sum()
        underrepresented_classes = counts.filter(nw.col("count") <= 0.1 * total)[
            value_col
        ].to_list()
        if len(underrepresented_classes) > 0:
            return (
                f"Classes {underrepresented_classes} each represent less than 10% of "
                "the dataset samples. Accuracy should not be used alone to assess "
                "model performance as it may be misleading by ignoring poor "
                "performance on underrepresented classes."
            )
        return None
