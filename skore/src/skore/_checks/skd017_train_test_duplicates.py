from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

from skore._checks.base import Check
from skore._checks.utils import (
    CheckNotApplicable,
    cast_report,
    get_report_X,
    get_report_y,
)

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckTrainTestDuplicates(Check):
    """Check for duplicate rows in the dataset before splitting (SKD017).

    Duplicate rows in the dataset leak across a train/test split: identical rows
    end up on both sides, so the model is evaluated on rows it was trained on and
    the reported score is overly optimistic. The check inspects the full, unsplit
    dataset (features and target) and reports how many rows are duplicated.
    """

    code = "SKD017"
    title = "Duplicate rows across train and test"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd017-train-test-duplicates"
    severity = "issue"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect duplicate rows in the full (unsplit) dataset."""
        report = cast_report(report)

        X = nw.from_native(get_report_X(report, data_source="both"), eager_only=True)
        y = nw.from_native(get_report_y(report, data_source="both"), allow_series=True)
        if isinstance(y, nw.Series):
            y = y.to_frame()
        data = nw.concat([X, y], how="horizontal")

        try:
            n_duplicated = sum(data.is_duplicated().to_list())
        except (TypeError, ValueError) as err:
            raise CheckNotApplicable("Could not compare rows for duplicates.") from err

        if n_duplicated > 0:
            percent = 100 * n_duplicated / len(data)
            return (
                f"{n_duplicated} row(s) ({percent:.1f}% of the dataset) are "
                "duplicated. Duplicate rows leak across a train/test split and "
                "inflate the reported score. If they are not legitimate, "
                "de-duplicate the dataset before splitting."
            )
        return None
