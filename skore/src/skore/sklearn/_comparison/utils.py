import copy
from typing import Optional

import pandas as pd

from skore.sklearn.types import Aggregate


def _combine_estimator_results(
    individual_results: list[pd.DataFrame],
    estimator_names: list[str],
    indicator_favorability: bool,
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
    """
    results = pd.concat(individual_results, axis=1)

    # Pop the favorability column if it exists, to:
    # - not use it in the aggregate operation
    # - later to only report a single column and not by split columns
    if indicator_favorability:
        favorability = results.pop("Favorability").iloc[:, 0]
    else:
        favorability = None

    results.columns = pd.Index(estimator_names, name="Estimator")

    if favorability is not None:
        results["Favorability"] = favorability

    return results


def _combine_cross_validation_results(
    individual_results: list[pd.DataFrame],
    estimator_names: list[str],
    indicator_favorability: bool,
    aggregate: Optional[Aggregate],
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
    """

    def add_model_name_to_index(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
        """Move the model name from the column index to the table index."""
        df = copy.copy(df)

        # Put the model name as a column
        df["Estimator"] = model_name

        # Put the model name into the index
        if "Label / Average" in df.index.names:
            new_index = ["Metric", "Label / Average", "Estimator"]
        else:
            new_index = ["Metric", "Estimator"]
        df = df.reset_index().set_index(new_index)

        # Then drop the model from the columns
        df.columns = df.columns.droplevel(0)

        return df

    def melt_df(df):
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

    results = individual_results.copy()

    # Pop the favorability column if it exists, to:
    # - not use it in the aggregate operation
    # - later to only report a single column and not by split columns
    if indicator_favorability:
        favorability = results[0]["Favorability"]
        for result in results:
            result.pop("Favorability")
    else:
        favorability = None

    dfs_model_name_in_index = [
        add_model_name_to_index(df, estimator_name)
        for df, estimator_name in zip(results, estimator_names)
    ]

    dfs_melted = [melt_df(df) for df in dfs_model_name_in_index]

    df = pd.concat(dfs_melted, axis=0)

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
        # Make sure the Split column goes [0, 1, 2, 0, 1, 2]
        # Rather than [0, 0, 1, 1, 2, 2]

        metric_order = df["Metric"].unique()

        df["metric_order_index"] = df["Metric"].apply(
            lambda x: list(metric_order).index(x)
        )

        if "Label / Average" in df.columns:
            by = ["metric_order_index", "Label / Average", "Estimator", "Split"]
        else:
            by = ["metric_order_index", "Estimator", "Split"]

        df = (
            df.sort_values(by=by)
            .drop("metric_order_index", axis=1)
            .reset_index(drop=True)
        )

        new_index = []
        for col in df.columns:
            if col == "Value":
                break
            new_index.append(col)
        df = df.set_index(new_index)

    if favorability is not None:
        df["Favorability"] = favorability

    return df
