from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import narwhals as nw

from skore._checks.base import Check
from skore._checks.utils import CheckNotApplicable, cast_report

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport
    from skore._reports.estimator.report import EstimatorReport


def _count_test_rows_in_train(
    X_train: nw.DataFrame,
    X_test: nw.DataFrame,
    y_train: Iterable | None,
    y_test: Iterable | None,
) -> int:
    """Count test rows whose features and label also appear in train.

    Comparison via ``is_duplicated`` treats NaN as equal; labels, when present,
    are included so rows with matching features but different labels don't count.
    """
    columns = X_train.columns
    if y_train is not None and y_test is not None:
        X_train = X_train.with_columns(
            __skd017_y__=nw.new_series(
                "__skd017_y__", list(y_train), backend=nw.get_native_namespace(X_train)
            )
        )
        X_test = X_test.with_columns(
            __skd017_y__=nw.new_series(
                "__skd017_y__", list(y_test), backend=nw.get_native_namespace(X_test)
            )
        )
        columns = X_train.columns

    train_unique = X_train.select(columns).unique()
    test_unique = X_test.select(columns).unique()

    combined = nw.concat([train_unique, test_unique], how="vertical")
    shared_mask = combined.is_duplicated().to_list()[len(train_unique) :]
    matching = test_unique[[i for i, shared in enumerate(shared_mask) if shared]]
    if len(matching) == 0:
        return 0

    # Map the matching unique rows back to the number of full test rows they cover.
    expanded = nw.concat([matching.unique(), X_test.select(columns)], how="vertical")
    expanded_mask = expanded.is_duplicated().to_list()[len(matching.unique()) :]
    return sum(expanded_mask)


class CheckTrainTestDuplicates(Check):
    """Check for duplicate rows across train and test (SKD017).

    Detects rows in the test set that are identical to training rows, which leaks
    training data into the evaluation. Raises :class:`CheckNotApplicable` when
    train data is unavailable or inputs are not narwhals DataFrames.
    """

    code = "SKD017"
    title = "Duplicate rows across train and test"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd017-train-test-duplicates"
    severity = "issue"
    slow = True

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

            X_test = X_test.select(X_train.columns)
            try:
                n_duplicates = _count_test_rows_in_train(
                    X_train, X_test, fold.y_train, fold.y_test
                )
            except (TypeError, ValueError) as err:
                raise CheckNotApplicable(
                    "Could not compare train and test rows for duplicates."
                ) from err

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
