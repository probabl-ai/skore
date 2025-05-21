"""Class definition of the metadata object, used in ``skore`` project."""

from __future__ import annotations

import typing

import pandas as pd

from .widget import ModelExplorerWidget

if typing.TYPE_CHECKING:
    from typing import Union

    from skore.sklearn import EstimatorReport


class Metadata(pd.DataFrame):
    """
    Metadata and metrics for all reports persisted in a project at a given moment.

    A metadata object is an extended ``pandas.DataFrame``, containing all the metadata
    and metrics of the reports that have been persisted in a project. It implements a
    custom HTML representation, that allows user to filter its reports using a parallel
    coordinates plot. In a Jupyter Notebook, this representation automatically replaces
    the standard ``pandas.DataFrame`` one and displays an interactive plot.

    The parallel coordinates plot is interactive, such the user can select a filter path
    and retrieve the corresponding reports.

    Outside a Jupyter Notebook, the metadata object can be used as a standard
    ``pandas.DataFrame`` object. This means that it can be questioned, divided etc.,
    using the standard ``pandas.DataFrame`` functions.

    Methods
    -------
    reports(filter=True) -> list[skore.sklearn.EstimatorReport]
        Return the reports referenced by the metadata object from the project.
        If a query string selection exists, it will be automatically applied before, to
        filter the reports to return. Otherwise, returns all reports.
    """

    _metadata = ["project"]

    @staticmethod
    def factory(project, /):
        """
        Construct a metadata object from ``project`` at a given moment.

        Parameters
        ----------
        project : Union[skore_local_project.Project, skore_hub_project.Project]
            The project from which the metadata object is to be constructed.

        Notes
        -----
        This function is not intended for direct use. Instead simply use the accessor
        ``skore.Project.reports.metadata``.
        """
        metadata = pd.DataFrame(project.reports.metadata(), copy=False)

        if not metadata.empty:
            metadata["learner"] = pd.Categorical(metadata["learner"])
            metadata.index = pd.MultiIndex.from_arrays(
                [
                    pd.RangeIndex(len(metadata)),
                    pd.Index(metadata.pop("id"), name="id", dtype=str),
                ]
            )

        # Cast standard dataframe to Metadata for lazy reports selection.
        metadata = Metadata(metadata, copy=False)
        metadata.project = project

        return metadata

    @property
    def _constructor(self) -> type[Metadata]:
        return Metadata

    def reports(self, *, filter: bool = True) -> list[EstimatorReport]:
        """
        Return the reports referenced by the metadata object from the project.

        Parameters
        ----------
        filter : bool, optional
            Filter the reports to return with the user query string selection, default
            True.
        """
        if self.empty:
            return []

        if not hasattr(self, "project") or "id" not in self.index.names:
            raise RuntimeError("Bad condition: it is not a valid `Metadata` object.")

        if filter and (querystr := self._query_string_selection()):
            self = self.query(querystr)

        return list(map(self.project.reports.get, self.index.get_level_values("id")))

    def _repr_html_(self):
        """Display the interactive plot and controls."""
        self._plot_widget = ModelExplorerWidget(dataframe=self)
        self._plot_widget.display()
        return ""

    def _query_string_selection(self) -> Union[str, None]:
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
