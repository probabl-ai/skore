from typing import Literal

from skore._sklearn._plot.style import StyleDisplayMixin
from skore._sklearn._plot.utils import HelpDisplayMixin
from skore._utils._index import flatten_multi_index


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
        scoring_names: Literal["verbose"] | dict[str, str] | None = None,
        indicator_favorability: bool = False,
        flat_index: bool = True,
    ):
        """Return the summarize as a dataframe.

        Parameters
        ----------
        scoring_names : {"verbose"} or dict, default=None
            Whether to override the default metric names.

            - if "verbose", the metric names are replaced by their verbose name
              capitalized and without underscores;
            - if a dictionary, the keys are the metric names and the values are the new
              names;
            - if `None`, the default metric names are used.

        indicator_favorability : bool, default=False
            Whether or not to add an indicator of the favorability of the metric as
            an extra column in the returned DataFrame.

        flat_index : bool, default=True
            Whether to return a flat index or a multi-index.

        Returns
        -------
        frame : pandas.DataFrame
            The report metrics as a dataframe.
        """
        df = self.data.dropna(axis="columns", how="all")

        if scoring_names == "verbose":
            df = df.drop(columns="metric").rename(columns={"verbose_name": "metric"})
            join_str = " "
        elif scoring_names is None:
            df = df.drop(columns="verbose_name")
            join_str = "_"
        elif isinstance(scoring_names, dict):
            df = df.drop(columns="verbose_name")
            df["metric"] = df["metric"].apply(lambda x: scoring_names.get(x, x))
            join_str = " "
        else:
            raise ValueError("`scoring_names` must be one of {'verbose', None, dict}.")

        if not indicator_favorability:
            df = df.drop(columns="favorability")

        for col in ("label", "output", "average"):
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].cat.add_categories([""])
                df[col] = df[col].fillna("")

        if self.report_type == "estimator":
            estimator_name = df.pop("estimator_name")
            index = df.columns.intersection(self._possible_index_columns).tolist()
            df = df.set_index(index)
            df = df.rename(columns={"score": estimator_name.cat.categories.item()})

        if flat_index:
            df.index = flatten_multi_index(df.index, join_str=join_str)
            df.index.name = "metric"

        return df

    @StyleDisplayMixin.style_plot
    def plot(self):
        raise NotImplementedError
