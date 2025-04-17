import pandas as pd

from .. import item as item_module
from ..client.client import AuthenticatedClient
from .widget import ModelExplorerWidget


class Metadata(pd.DataFrame):
    _metadata = ["project"]

    @staticmethod
    def factory(project, /):
        def dto(summary):
            return dict(
                (
                    ("id", summary["id"]),
                    ("run_id", summary["run_id"]),
                    ("ml_task", summary["ml_task"]),
                    ("learner", summary["estimator_class_name"]),
                    ("dataset", summary["dataset_fingerprint"]),
                    ("date", summary["created_at"]),
                    *(
                        (metric["name"], metric["value"])
                        for metric in summary["metrics"]
                        if metric["data_source"] in (None, "test")
                    ),
                )
            )

        # Retrieve HUB's metadata
        with AuthenticatedClient(raises=True) as client:
            response = client.get(
                "/".join(
                    (
                        "projects",
                        project.tenant,
                        project.name,
                        "experiments",
                        "estimator-reports",
                    )
                )
            )

        summaries = response.json()

        if not summaries:
            raise Exception

        # Process the HUB's metadata to be usable by the widget
        summaries = pd.DataFrame(map(dto, summaries), copy=False)
        summaries["learner"] = pd.Categorical(summaries["learner"])
        summaries.index = pd.MultiIndex.from_arrays(
            [
                pd.RangeIndex(len(summaries)),
                pd.Index(summaries.pop("id"), name="id", dtype=str),
            ]
        )

        # Cast standard dataframe to Metadata for lazy reports selection.
        metadata = Metadata(summaries, copy=False)
        metadata.project = project

        return metadata

    @property
    def _constructor(self):
        return Metadata

    def reports(self):
        if not hasattr(self, "project") or "id" not in self.index.names:
            raise Exception

        def dto(response):
            report = response.json()
            item_class_name = report["raw"]["class"]
            item_class = getattr(item_module, item_class_name)
            item_parameters = report["raw"]["parameters"]
            item = item_class(**item_parameters)
            return item.__raw__

        ids = list(self.index.get_level_values("id"))

        with AuthenticatedClient(raises=True) as client:
            return [
                dto(
                    client.get(
                        "/".join(
                            (
                                "projects",
                                self.project.tenant,
                                self.project.name,
                                "experiments",
                                "estimator-reports",
                                id,
                            )
                        )
                    )
                )
                for id in ids
            ]

    def _repr_html_(self):
        """Display the interactive plot and controls."""
        self._plot_widget = ModelExplorerWidget(dataframe=self)
        self._plot_widget.display()

    def query_string_selection(self) -> str:
        """Generate a pandas query string based on user selections in the plot.

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
        >>> query_string = df.query_string_selection()
        >>> df_filtered = df.query(query_string)
        """
        selection = self._plot_widget.current_selection.copy()
        query_parts = []

        task = selection.pop("ml_task")
        query_parts.append(f"ml_task == '{task}'")

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

        # Join all query parts with logical AND
        if query_parts:
            return " and ".join(query_parts)
        else:
            return ""
