from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

from skore._checks.base import Check
from skore._checks.utils import CheckNotApplicable, cast_report

if TYPE_CHECKING:
    from skore._reports.base import _BaseReport
    from skore._reports.cross_validation.report import CrossValidationReport
    from skore._reports.estimator.report import EstimatorReport


def _full_dataset(
    report: EstimatorReport | CrossValidationReport,
) -> tuple[nw.DataFrame, list | None]:
    """Return the full (unsplit) feature matrix and target of the report.

    For cross-validation reports the data is already unsplit (``report.X`` /
    ``report.y``). For estimator reports, train and test are re-concatenated so
    the check runs once on the data as it was before splitting.
    """
    if report._report_type == "cross-validation":
        if not nw.dependencies.is_into_dataframe(report.X):
            raise CheckNotApplicable(
                "Input data is not a narwhals compatible DataFrame. "
                f"Got {type(report.X).__name__}."
            )
        X = nw.from_native(report.X, eager_only=True)
        return X, (None if report.y is None else list(report.y))

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

    X_train = nw.from_native(report.X_train, eager_only=True)
    X_test = nw.from_native(report.X_test, eager_only=True)
    if sorted(X_train.columns) != sorted(X_test.columns):
        raise CheckNotApplicable(
            "Train and test feature matrices have different columns."
        )
    X = nw.concat([X_train, X_test.select(X_train.columns)], how="vertical")

    if report.y_train is None or report.y_test is None:
        return X, None
    y = list(report.y_train) + list(report.y_test)
    return X, y


class CheckTrainTestDuplicates(Check):
    """Check for duplicate rows in the dataset before splitting (SKD017).

    The only way a row ends up in both the train and the test set after a
    random split is if the dataset contains duplicate rows in the first place.
    This check therefore inspects the full, unsplit dataset and flags duplicate
    rows (identical features and, when available, identical label). Splitting
    such a dataset without de-duplication is likely to leak training rows into
    the test set.

    Raises :class:`CheckNotApplicable` when train data is unavailable or inputs
    are not narwhals DataFrames.
    """

    code = "SKD017"
    title = "Duplicate rows across train and test"
    report_types = ["estimator", "cross-validation"]
    docs_url = "skd017-train-test-duplicates"
    severity = "issue"
    slow = True

    def check_function(self, report: _BaseReport) -> str | None:
        """Detect duplicate rows in the full (unsplit) dataset."""
        input_report = cast_report(report)

        X, y = _full_dataset(input_report)

        if y is not None:
            X = X.with_columns(
                __skd017_y__=nw.new_series(
                    "__skd017_y__", y, backend=nw.get_native_namespace(X)
                )
            )

        try:
            # ``is_duplicated`` marks every row that has at least one identical
            # twin elsewhere in the dataset (NaN compares equal).
            duplicated_mask = X.is_duplicated().to_list()
        except (TypeError, ValueError) as err:
            raise CheckNotApplicable("Could not compare rows for duplicates.") from err

        n_duplicated = sum(duplicated_mask)
        if n_duplicated > 0:
            percent = 100 * n_duplicated / len(X)
            return (
                f"{n_duplicated} row(s) ({percent:.1f}% of the dataset) are "
                "duplicates of another row. Splitting a dataset with duplicate "
                "rows without de-duplication leaks training rows into the test "
                "set and inflates the reported score; de-duplicate the dataset "
                "before splitting, or use a grouped splitter so identical rows "
                "stay on the same side of the split."
            )
        return None
