"""Class definition of the summary object, used in ``skore`` project."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import Categorical, DataFrame, Index, MultiIndex, RangeIndex

from skore import ComparisonReport
from skore.project.widget import ModelExplorerWidget

if TYPE_CHECKING:
    from typing import Literal

    from skore import CrossValidationReport, EstimatorReport


class Summary(DataFrame):
    """
    Metadata and metrics for all reports persisted in a project at a given moment.

    A summary object is an extended :class:`pandas.DataFrame`, containing all the
    metadata and metrics of the reports that have been persisted in a project. It
    implements a custom HTML representation, that allows user to filter its reports
    using a parallel coordinates plot. In a Jupyter Notebook, this representation
    automatically replaces the standard :class:`pandas.DataFrame` one and displays an
    interactive plot.

    The parallel coordinates plot is interactive, such the user can select a filter path
    and retrieve the corresponding reports.

    Outside a Jupyter Notebook, the summary object can be used as a standard
    :class:`pandas.DataFrame` object. This means that it can be questioned, divided
    etc., using the standard :class:`pandas.DataFrame` functions.

    Refer to :class:`pandas.DataFrame` for the standard methods and attributes.
    """

    _metadata = ["project"]

    @staticmethod
    def factory(project, /):
        """
        Construct a summary object from ``project`` at a given moment.

        Parameters
        ----------
        project : ``skore_local_project.Project`` | ``skore_hub_project.Project``
            The project from which the summary object is to be constructed.

        Notes
        -----
        This function is not intended for direct use. Instead simply use the accessor
        :meth:`skore.Project.summarize`.
        """
        summary = DataFrame(project.summarize(), copy=False)

        if not summary.empty:
            summary["learner"] = Categorical(summary["learner"])
            summary.index = MultiIndex.from_arrays(
                [
                    RangeIndex(len(summary)),
                    Index(summary.pop("id"), name="id", dtype=str),
                ]
            )

        # Cast standard dataframe to Summary for lazy reports selection.
        summary = Summary(summary, copy=False)
        summary.project = project

        return summary

    @property
    def _constructor(self) -> type[Summary]:
        return Summary

    def reports(
        self,
        *,
        filter: bool = True,
        return_as: Literal["list", "comparison"] = "list",
    ) -> list[EstimatorReport | CrossValidationReport] | ComparisonReport:
        """
        Return the reports referenced by the summary object from the project.

        Parameters
        ----------
        filter : bool, optional
            Filter the reports to return with the user query string selection, default
            True.
        return_as : Literal["list", "comparison"], optional
            Return reports as flat list or comparison report, default list.
        """
        if self.empty:
            return []

        if not hasattr(self, "project") or "id" not in self.index.names:
            raise RuntimeError("Bad condition: it is not a valid `Summary` object.")

        if filter and (querystr := self._query_string_selection()):
            self = self.query(querystr)

        reports = [self.project.get(id) for id in self.index.get_level_values("id")]

        if return_as == "comparison":
            try:
                return ComparisonReport(reports)
            except ValueError as e:
                raise RuntimeError(
                    f"Bad condition: the comparison mode is only applicable when "
                    f"reports have the same dataset.\n"
                    f"Found '{self['dataset'].unique()}'.\n"
                    f"Please query the dataframe or use the widget to make your "
                    f"selection."
                ) from e
        return reports

    # Override pandas DataFrame's _repr_html_ to only show the widget
    def _repr_html_(self):
        return ""

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Display the interactive plot and controls."""
        self._plot_widget = ModelExplorerWidget(dataframe=self)
        return {"text/html": self._plot_widget.display()}

    def _query_string_selection(self) -> str | None:
        """
        Generate a pandas query string based on user selections in the plot.

        This method translates the visual selections made on the parallel
        coordinates plot into a pandas query string that can be used to
        filter the dataframe.

        Returns
        -------
        str
            A query string that can be used with DataFrame.query() to filter
            the original dataframe based on the visual selection.
            Returns an empty string if no selections are active.

        Examples
        --------
        >>> # xdoctest: +SKIP
        >>> query_string = df._query_string_selection()
        >>> df_filtered = df.query(query_string)
        """
        if not hasattr(self, "_plot_widget"):
            return None

        self._plot_widget.update_selection()
        selection = self._plot_widget.current_selection.copy()
        query_parts = []

        task = selection.pop("ml_task")
        query_parts.append(f"ml_task.str.contains('{task}')")

        dataset = selection.pop("dataset")
        query_parts.append(f"dataset == '{dataset}'")

        for column_name, range_values in selection.items():
            if not isinstance(range_values[0], tuple):  # single selection
                range_values = (range_values,)

            if column_name == "learner":
                for dim in self._plot_widget.current_fig.data[0].dimensions:
                    if dim["label"] == "Learner":
                        learner_values = dim["ticktext"]
                        learner_codes = dim["tickvals"]
                        break

                selected_learners = []
                for min_val, max_val in range_values:
                    # When selecting on the parallel coordinates plot, the min and max
                    # values my not be exactly the min or max of the learner codes. We
                    # clip them to be sure that it falls into the correct range.
                    if min_val < 1e-8:
                        min_val = 0
                    if max_val > max(learner_codes) - 1e-8:
                        max_val = max(learner_codes)

                    for i, learner in enumerate(learner_values):
                        # Get the jittered code value for this learner
                        code_val = learner_codes[i]
                        if min_val <= code_val <= max_val:
                            selected_learners.append(learner)

                if selected_learners:
                    values_str = ", ".join(
                        [f"'{value}'" for value in selected_learners]
                    )
                    query_parts.append(f"learner.isin([{values_str}])")
            else:
                dim_query = []
                for min_val, max_val in range_values:
                    dim_query.append(
                        f"({column_name} >= {min_val:.6f} and "
                        f"{column_name} <= {max_val:.6f})"
                    )
                query_parts.append("(" + " or ".join(dim_query) + ")")

        return " and ".join(query_parts)
