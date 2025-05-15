from __future__ import annotations

import typing

import pandas as pd

from .widget import ModelExplorerWidget

if typing.TYPE_CHECKING:
    from typing import Union


class Metadata(pd.DataFrame):
    _metadata = ["project"]

    @staticmethod
    def factory(project, /):
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
    def _constructor(self):
        return Metadata

    def reports(self, *, filter=True):
        """"""
        if not hasattr(self, "project") or "id" not in self.index.names:
            raise RuntimeError("Bad condition: it is not a valid `Metadata` object.")

        if filter and (querystr := self.query_string_selection()):
            self = self.query(querystr)

        return list(map(self.project.reports.get, self.index.get_level_values("id")))

    def _repr_html_(self):
        """Display the interactive plot and controls."""
        self._plot_widget = ModelExplorerWidget(dataframe=self)
        self._plot_widget.display()
        return ""

    def query_string_selection(self) -> Union[str, None]:
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
        >>> query_string = df.query_string_selection()
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
            if column_name == "learner":
                learner_values = self["learner"].cat.categories
                learner_codes = self["learner"].cat.codes

                selected_learners = []
                min_val, max_val = range_values

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
                min_val, max_val = range_values
                query_parts.append(
                    f"({column_name} >= {min_val:.6f} and "
                    f"{column_name} <= {max_val:.6f})"
                )

        return " and ".join(query_parts)
