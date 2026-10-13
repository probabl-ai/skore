from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

from skore._checks.base import Check
from skore._checks.utils import CheckNotApplicable, cast_report

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport


class CheckTrainTestTimeOverlap(Check):
    """Check for train/test temporal overlap (SKD013).

    Flags datetime columns where the latest train timestamp is at or after
    the earliest test timestamp, indicating that future points leak into
    the training set (e.g. data was shuffled before splitting a time series).

    Raises :class:`CheckNotApplicable` when train and test inputs are not
    pandas DataFrames or contain no datetime column.
    """

    code = "SKD013"
    title = "Train-test overlap in time series"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd013-train-test-time-overlap"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        input_report = cast_report(report)

        overlapping: set[str] = set()
        found_datetime = False
        for report in (
            input_report.reports_
            if input_report._report_type == "cross-validation"
            else [input_report]
        ):
            if report.X_train is None:
                raise CheckNotApplicable("Train data is unavailable.")
            if not nw.dependencies.is_into_dataframe(report.X_train):
                raise CheckNotApplicable(
                    "Input data is not a narwhals compatible DataFrame. "
                    f"Got {type(report.X_train).__name__}."
                )
            if not nw.dependencies.is_into_dataframe(report.X_test):
                raise CheckNotApplicable(
                    "Input data is not a narwhals compatible DataFrame. "
                    f"Got {type(report.X_test).__name__}."
                )
            X_train_nw = nw.from_native(report.X_train)
            X_test_nw = nw.from_native(report.X_test)

            datetime_columns = sorted(
                set(X_train_nw.select(nw.selectors.datetime()).columns)
                & set(X_test_nw.select(nw.selectors.datetime()).columns)
            )
            if datetime_columns:
                found_datetime = True
                overlapping.update(
                    col
                    for col in datetime_columns
                    if X_train_nw[col].max() >= X_test_nw[col].min()
                )

        if not found_datetime:
            raise CheckNotApplicable("No datetime column found.")
        if overlapping:
            return (
                f"Datetime column(s) {sorted(overlapping)} contain training "
                "timestamps that are after the earliest test timestamp. Future "
                "points may be leaking into the training set; consider a "
                "time-based split."
            )
        return None
