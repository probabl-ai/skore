import copy
from typing import Literal

import pandas as pd

from skore._sklearn.types import Aggregate, DataSource


def _combine_estimator_results(
    individual_results: list[pd.DataFrame],
    estimator_names: list[str],
    indicator_favorability: bool,
    data_source: DataSource | Literal["both"],
) -> pd.DataFrame:
    """Combine a list of dataframes provided by `EstimatorReport`s.

    Parameters
    ----------
    results : pd.DataFrame
        The dataframes to combine.
        They are assumed to originate from a `EstimatorReport.metrics` computation.

    estimator_names : list of str of len (len(results))
        The name to give the estimator for each dataframe.

    indicator_favorability : bool
        Whether to keep the Favorability column.

    data_source : {"test", "train", "X_y", "both"}
        The data source.

    Examples
    --------
    >>> from skore._sklearn._comparison.utils import _combine_estimator_results
    >>> import pandas as pd
    >>> individual_results = [
    ...     pd.DataFrame.from_dict(
    ...         {
    ...             "index": ["Brier score"],
    ...             "columns": ["LogisticRegression"],
    ...             "data": [[0.026]],
    ...             "index_names": ["Metric"],
    ...             "column_names": [None],
    ...         },
    ...         orient="tight",
    ...     ),
    ...     pd.DataFrame.from_dict(
    ...         {
    ...             "index": ["Brier score"],
    ...             "columns": ["LogisticRegression"],
    ...             "data": [[0.026]],
    ...             "index_names": ["Metric"],
    ...             "column_names": [None],
    ...         },
    ...         orient="tight",
    ...     ),
    ... ]
    >>> estimator_names = ['LogisticRegression_1', 'LogisticRegression_2']
    >>> _combine_estimator_results(
    ...     individual_results,
    ...     estimator_names,
    ...     indicator_favorability=False,
    ...     data_source="test",
    ... )
    Estimator    LogisticRegression_1  LogisticRegression_2
    Metric
    Brier score                   ...                   ...
    """
    results = pd.concat(individual_results, axis=1)

    # Pop the favorability column if it exists, to:
    # - not use it in the aggregate operation
    # - later to only report a single column and not by split columns
    if indicator_favorability:
        # Some metrics can be undefined for some estimators and NaN are
        # introduced after the concatenation. We fill the NaN using the
        # valid favorability
        favorability = results.pop("Favorability").bfill(axis=1).iloc[:, 0]
    else:
        favorability = None

    if data_source == "both":
        results.columns = pd.Index(
            [
                x
                for name in estimator_names
                for x in [f"{name} (train)", f"{name} (test)"]
            ],
            name="Estimator",
        )
    else:
        results.columns = pd.Index(estimator_names, name="Estimator")

    if favorability is not None:
        results["Favorability"] = favorability

    return results


def _combine_cross_validation_results(
    individual_results: list[pd.DataFrame],
    estimator_names: list[str],
    indicator_favorability: bool,
    aggregate: Aggregate | None,
) -> pd.DataFrame:
    """Combine a list of dataframes provided by `CrossValidationReport`s.

    Parameters
    ----------
    results : pd.DataFrame
        The dataframes to combine.
        They are assumed to originate from a `CrossValidationReport.metrics`
        computation. In particular, there are several assumptions made:

        - every dataframe has the form:
            - index: Index "Metric", or MultiIndex ["Metric", "Label / Average"]
            - columns: MultiIndex with levels ["Estimator", "Splits"] (can be
              unnamed)
        - all dataframes have the same metrics

        The dataframes are not required to have the same number of columns (splits).

    estimator_names : list of str of length `len(results)`
        The name to give the estimator for each dataframe.

    indicator_favorability : bool
        Whether to keep the Favorability column.

    aggregate : Aggregate
        How to aggregate the resulting dataframe.

    Examples
    --------
    >>> from skore._sklearn._comparison.utils import _combine_cross_validation_results
    >>> import pandas as pd
    >>> pd.set_option('display.max_columns', None)
    >>> pd.set_option('display.width', 1000)
    >>> individual_results = [
    ...     pd.DataFrame.from_dict(
    ...         {
    ...             "index": ["Accuracy"],
    ...             "columns": [
    ...                 ("DummyClassifier", "Split #0"),
    ...                 ("DummyClassifier", "Split #1"),
    ...                 ("DummyClassifier", "Split #2"),
    ...                 ("DummyClassifier", "Split #3"),
    ...                 ("DummyClassifier", "Split #4"),
    ...             ],
    ...             "data": [[0.45, 0.45, 0.35, 0.35, 0.55]],
    ...             "index_names": ["Metric"],
    ...             "column_names": [None, None],
    ...         },
    ...         orient="tight",
    ...     ),
    ...     pd.DataFrame.from_dict(
    ...         {
    ...             "index": ["Accuracy"],
    ...             "columns": [
    ...                 ("DummyClassifier", "Split #0"),
    ...                 ("DummyClassifier", "Split #1"),
    ...                 ("DummyClassifier", "Split #2"),
    ...             ],
    ...             "data": [[0.53, 0.42, 0.52]],
    ...             "index_names": ["Metric"],
    ...             "column_names": [None, None],
    ...         },
    ...         orient="tight",
    ...     ),
    ... ]
    >>> estimator_names = ["DummyClassifier_1", "DummyClassifier_2"]
    >>> _combine_cross_validation_results(
    ...     individual_results,
    ...     estimator_names,
    ...     indicator_favorability=False,
    ...     aggregate=None,
    ... )
                                            Value
    Metric   Estimator         Split
    Accuracy DummyClassifier_1 Split #0       ...
                               Split #1       ...
                               Split #2       ...
                               Split #3       ...
                               Split #4       ...
             DummyClassifier_2 Split #0       ...
                               Split #1       ...
                               Split #2       ...
    >>> _combine_cross_validation_results(
    ...     individual_results,
    ...     estimator_names,
    ...     indicator_favorability=False,
    ...     aggregate=["mean", "std"],
    ... )
                           mean                                 std
    Estimator DummyClassifier_1 DummyClassifier_2 DummyClassifier_1 DummyClassifier_2
    Metric
    Accuracy                ...               ...               ...               ...
    """

    def df_with_name(df: pd.DataFrame, estimator_name: str) -> pd.DataFrame:
        """Move the estimator name from the column index to the table index.

        Notes
        -----
        This mutates the input DataFrame.
        """
        df["Estimator"] = estimator_name

        if "Label / Average" in df.index.names:
            new_index = ["Metric", "Label / Average", "Estimator"]
        else:
            new_index = ["Metric", "Estimator"]
        df = df.reset_index().set_index(new_index)

        # Then drop the `estimator_name` from the columns index
        df.columns = df.columns.droplevel(0)

        return df

    def melt(df):
        """Move split number from columns to index."""
        df_reset = df.reset_index()

        if "Label / Average" in df_reset.columns:
            id_vars = ["Metric", "Label / Average", "Estimator"]
        else:
            id_vars = ["Metric", "Estimator"]

        melted = pd.melt(
            df_reset,
            id_vars=id_vars,
            var_name="Split",
            value_name="Value",
        )

        return melted

    def sort_by_split(df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe, preserving the order of metrics.

        Ensures that the Split column goes [0, 1, 2, 0, 1, 2] rather than
        [0, 0, 1, 1, 2, 2].

        Notes
        -----
        Mutates the input dataframe.

        Examples
        --------
        >>> df = pd.DataFrame.from_dict(
        ...     {
        ...         "index": range(8),
        ...         "columns": [
        ...             "Metric",
        ...             "Label / Average",
        ...             "Estimator",
        ...             "Split",
        ...             "Value",
        ...         ],
        ...         "data": [
        ...             ["Precision", 0, "DummyClassifier_1", "Split #0", 0.44],
        ...             ["Precision", 1, "DummyClassifier_1", "Split #0", 0.45],
        ...             ["Precision", 0, "DummyClassifier_1", "Split #1", 0.44],
        ...             ["Precision", 1, "DummyClassifier_1", "Split #1", 0.45],
        ...             ["Precision", 0, "DummyClassifier_2", "Split #0", 0.53],
        ...             ["Precision", 1, "DummyClassifier_2", "Split #0", 0.52],
        ...             ["Precision", 0, "DummyClassifier_2", "Split #1", 0.42],
        ...             ["Precision", 1, "DummyClassifier_2", "Split #1", 0.42],
        ...         ],
        ...         "index_names": [None],
        ...         "column_names": [None],
        ...     },
        ...     orient="tight",
        ... )
        >>> sort_by_split(df)
              Metric  Label / Average          Estimator     Split  Value
        0  Precision                0  DummyClassifier_1  Split #0   0.44
        1  Precision                0  DummyClassifier_1  Split #1   0.44
        2  Precision                0  DummyClassifier_2  Split #0   0.53
        3  Precision                0  DummyClassifier_2  Split #1   0.42
        4  Precision                1  DummyClassifier_1  Split #0   0.45
        5  Precision                1  DummyClassifier_1  Split #1   0.45
        6  Precision                1  DummyClassifier_2  Split #0   0.52
        7  Precision                1  DummyClassifier_2  Split #1   0.42
        """
        df["Metric"] = df["Metric"].astype(
            pd.CategoricalDtype(df["Metric"].unique(), ordered=True)
        )
        df["metric_order_index"] = df["Metric"].cat.codes

        if "Label / Average" in df.columns:
            by = ["metric_order_index", "Label / Average", "Estimator", "Split"]
        else:
            by = ["metric_order_index", "Estimator", "Split"]

        df = (
            df.sort_values(by=by)
            .drop("metric_order_index", axis=1)
            .reset_index(drop=True)
        )

        df["Metric"] = df["Metric"].astype(str)

        return df

    # Deepcopy the contained dataframes to avoid mutating them;
    # in particular, they might be provided by a report cache
    results = copy.deepcopy(individual_results)

    # Pop the favorability column if it exists, to:
    # - not use it in the aggregate operation
    # - later to only report a single column and not by split columns
    if indicator_favorability:
        # Some metrics can be undefined for some estimators and NaN are
        # introduced after the concatenation. We fill the NaN using the
        # valid favorability
        favorability = (
            pd.concat([result.pop("Favorability") for result in results], axis=1)
            .bfill(axis=1)
            .iloc[:, 0]
        )
    else:
        favorability = None

    df = pd.concat(
        [
            melt(df_with_name(df, estimator_name))
            for df, estimator_name in zip(results, estimator_names, strict=False)
        ],
        axis=0,
    )

    if aggregate:
        if isinstance(aggregate, str):
            aggregate = [aggregate]

        if "Label / Average" in df.columns:
            index = ["Metric", "Label / Average"]
        else:
            index = ["Metric"]

        df = df.pivot_table(
            index=index,
            columns="Estimator",
            values="Value",
            aggfunc=aggregate,
            sort=False,
        )
    else:
        df = sort_by_split(df)
        df = df.set_index(list(df.columns.drop("Value")))

    if favorability is not None:
        df["Favorability"] = favorability

    return df
