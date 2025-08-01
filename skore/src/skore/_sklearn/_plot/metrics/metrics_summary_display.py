from typing import Literal

import pandas as pd

from skore._sklearn._plot.style import StyleDisplayMixin
from skore._sklearn._plot.utils import HelpDisplayMixin
from skore._sklearn.types import Aggregate
from skore._utils._index import flatten_multi_index, transform_index


class MetricsSummaryDisplay(HelpDisplayMixin, StyleDisplayMixin):
    """Display for summarize.

    An instance of this class will be created by `Report.metrics.summarize()`.
    This class should not be instantiated directly.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to display.

    report_type : {"estimator", "comparison-estimator", "cross-validation", \
            "comparison-cross-validation"}
        The type of report.
    """

    _possible_index_columns: set[str] = {"metric", "label", "output", "average"}

    def __init__(self, data, report_type):
        self.data = data
        self.report_type = report_type

    def frame(
        self,
        *,
        scoring_names: dict[str, str] | None = None,
        aggregate: Aggregate | None = ("mean", "std"),
        indicator_favorability: bool = False,
        flat_index: bool = True,
        case: Literal["pretty", "snake"] = "snake",
    ):
        """Return the summarize as a dataframe.

        Parameters
        ----------
        scoring_names : dict, default=None
            Whether to override the default metric names. Pass a dictionary where the
            keys are the metric names and the values are the new names.

        aggregate : {"mean", "std"}, list of such str or None, default=("mean", "std")
            Only used when `report_type` is `"cross-validation"`.
            Function to aggregate the scores across the cross-validation splits.
            None will return the scores for each split.

        indicator_favorability : bool, default=False
            Whether or not to add an indicator of the favorability of the metric as
            an extra column in the returned DataFrame.

        flat_index : bool, default=True
            Whether to return a flat index or a multi-index.

        case : {"pretty", "snake"}, default="snake"
            Whether to use the pretty (i.e. capitalize and white space separated) or
            snake case (i.e. lower case and underscore separated) format for the metric
            and column names.

        Returns
        -------
        frame : pandas.DataFrame
            The report metrics as a dataframe.
        """
        df = self.data.dropna(axis="columns", how="all").drop(columns="verbose_name")

        if isinstance(scoring_names, dict):
            df["metric"] = df["metric"].apply(lambda x: scoring_names.get(x, x))
        elif scoring_names is not None:
            raise ValueError("`scoring_names` must be a dictionary or None.")

        if case not in ("pretty", "snake"):
            raise ValueError("`case` must be one of {'pretty', 'snake'}.")

        aggregate = [aggregate] if isinstance(aggregate, str) else aggregate

        for col in ("label", "output", "average"):
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].cat.add_categories([""])
                df[col] = df[col].fillna("")

        if self.report_type == "estimator":
            estimator_name = df.pop("estimator_name")
            index = df.columns.intersection(self._possible_index_columns).tolist()
            df = df.set_index(index)
            df = df.rename(columns={"score": estimator_name.cat.categories.item()})
            if not indicator_favorability:
                df = df.drop(columns="favorability")
        elif self.report_type == "cross-validation":
            estimator_name = df.pop("estimator_name")
            index = df.columns.intersection(self._possible_index_columns).tolist()
            df = df.rename(columns={"score": estimator_name.cat.categories.item()})

            # Get the favorability column with the right index for later concatenation.
            # Since we have several splits, the favorability is repeated and we
            # only preserve the first occurrence.
            favorability = df.set_index(index).pop("favorability")
            favorability = favorability[
                ~favorability.index.duplicated(keep="first")
            ].to_frame()

            if aggregate is None:
                df = df.drop(columns="favorability").pivot(
                    index=index, columns="split_index"
                )
            else:
                df = (
                    df.drop(columns=["split_index", "favorability"])
                    .groupby(index, observed=True)
                    .agg(aggregate)
                )

            # Use the original ordering present in the favorability index that has been
            # lost due to pivoting or groupby.
            df = df.reindex(favorability.index)

            if indicator_favorability:
                # Convert index to multi-index with an empty level corresponding to
                # the split index.
                favorability.columns = pd.MultiIndex.from_tuples(
                    [
                        ("favorability", ""),
                    ],
                    names=df.columns.names,
                )
                df = pd.concat([df, favorability], axis=1)

        if flat_index:
            join_str = " " if case == "pretty" else "_"
            df.index = flatten_multi_index(df.index, join_str=join_str)
            df.index.name = "metric"
            df.columns = flatten_multi_index(df.columns, join_str=join_str)

        df.index = transform_index(df.index, case_type=case)
        df.columns = transform_index(df.columns, case_type=case)

        return df

    @StyleDisplayMixin.style_plot
    def plot(self):
        raise NotImplementedError
