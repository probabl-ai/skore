from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

from skore._checks.base import Check
from skore._checks.utils import CheckNotApplicable, cast_report

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport
    from skore._reports.estimator.report import EstimatorReport


class CheckTrainTestDuplicates(Check):
    """Check for duplicate rows shared between train and test (SKD017).

    Flags rows present in ``X_test`` that are identical to rows in ``X_train``.
    Such overlap leaks training data into the evaluation set and inflates the
    reported score.

    Raises :class:`CheckNotApplicable` when train data is unavailable or when
    the inputs are not narwhals-compatible DataFrames.
    """

    code = "SKD017"
    title = "Duplicate rows across train and test"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd017-train-test-duplicates"
    severity = "issue"

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect rows shared between the train and test feature matrices."""
        input_report = cast_report(report)

        reports: list[EstimatorReport] = (
            input_report.reports_
            if input_report._report_type == "cross-validation"
            else [input_report]  # type: ignore[list-item]
        )

        total_duplicates = 0
        total_test = 0
        for fold in reports:
            if fold.X_train is None:
                raise CheckNotApplicable("Train data is unavailable.")
            if not nw.dependencies.is_into_dataframe(fold.X_train):
                raise CheckNotApplicable(
                    "Input data is not a narwhals compatible DataFrame. "
                    f"Got {type(fold.X_train).__name__}."
                )
            if not nw.dependencies.is_into_dataframe(fold.X_test):
                raise CheckNotApplicable(
                    "Input data is not a narwhals compatible DataFrame. "
                    f"Got {type(fold.X_test).__name__}."
                )

            X_train = nw.from_native(fold.X_train, eager_only=True)
            X_test = nw.from_native(fold.X_test, eager_only=True)

            if sorted(X_train.columns) != sorted(X_test.columns):
                raise CheckNotApplicable(
                    "Train and test feature matrices have different columns."
                )

            columns = X_train.columns
            train_keys = {tuple(row) for row in X_train.select(columns).iter_rows()}
            n_duplicates = sum(
                tuple(row) in train_keys for row in X_test.select(columns).iter_rows()
            )

            total_duplicates += n_duplicates
            total_test += len(X_test)

        if total_duplicates > 0:
            percent = 100 * total_duplicates / total_test
            return (
                f"{total_duplicates} test row(s) ({percent:.1f}% of the test set) are "
                "identical to rows in the training set. Duplicate rows shared between "
                "train and test leak training data into the evaluation and inflate the "
                "reported score; de-duplicate the dataset before splitting."
            )
        return None
