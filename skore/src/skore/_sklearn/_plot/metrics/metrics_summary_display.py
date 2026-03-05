import pandas as pd

from skore._sklearn._plot.base import DisplayMixin
from skore._sklearn.types import ReportType
from skore._utils._index import flatten_multi_index


class MetricsSummaryDisplay(DisplayMixin):
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

    def __init__(
        self,
        data: pd.DataFrame,
        report_type: ReportType,
    ):
        self.data = data
        self.report_type = report_type

    def frame(
        self,
        *,
        favorability: bool = False,
        flat_index: bool = False,
    ):
        """Return the summarize as a dataframe.

        Parameters
        ----------
        favorability : bool, default=False
            Whether or not to add an indicator of the favorability of the metric as
            an extra column in the returned DataFrame.

        flat_index : bool, default=False
            Whether to return a flat index or a multi-index.

        Returns
        -------
        frame : pandas.DataFrame
            The report metrics as a dataframe.
        """
        df = self.data.copy()

        if self.report_type == "estimator":
            df = df.dropna(axis="columns", how="all")

            for col in df.columns.intersection(["label", "output", "average"]):
                df[col] = df[col].astype("str").replace("<NA>", "").fillna("")

            estimator_name = df.pop("estimator_name")[0]
            index = df.columns.intersection(
                ["metric", "verbose_name", "label", "output", "average"]
            ).to_list()
            df = df.set_index(index)

            if not favorability:
                df = df.drop(columns="favorability")
            else:
                # Put favorability at the end
                df = df[
                    [col for col in df.columns if col != "favorability"]
                    + ["favorability"]
                ]

            # Rename columns as well as index names
            new_columns = {
                "metric": "Metric",
                "verbose_name": "Metric",
                "label": "Label / Average",
                "output": "Output",
                "average": "Average",
                "favorability": "Favorability",
                "score": estimator_name,
            }
            df = df.rename(columns=new_columns)
            df.index = df.index.set_names(
                [new_columns.get(name, name) for name in df.index.names]
            )

            if df["data_source"].nunique() == 1:
                df = df.drop(columns="data_source")
            else:
                # Show metrics one column per data source
                df_pivoted = df.reset_index().pivot_table(
                    index=df.index.names,
                    columns="data_source",
                    values=estimator_name,
                    sort=False,
                )
                df_pivoted.columns = [
                    f"{estimator_name} ({col})" for col in df_pivoted.columns
                ]

                if favorability:
                    df_pivoted["Favorability"] = df[df["data_source"] == "test"][
                        "Favorability"
                    ]

                df = df_pivoted.copy()

            # Apply flat_index transformation after pivot (if needed)
            if flat_index:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = flatten_multi_index(df.columns)
                if isinstance(df.index, pd.MultiIndex):
                    df.index = flatten_multi_index(df.index)
                if isinstance(df.index, pd.Index):
                    df.index = df.index.str.replace(r"\((.*)\)$", r"\1", regex=True)

        return df

    @DisplayMixin.style_plot
    def plot(self):
        """Not yet implemented."""
        raise NotImplementedError
