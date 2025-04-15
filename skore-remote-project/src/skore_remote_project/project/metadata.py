import pandas as pd

from .. import item as item_module
from ..client.client import AuthenticatedClient
from .widget import ModelExplorerWidget

DIMENSION_TO_COLUMN = {
    "Dataset": "dataset",
    "Learner": "learner",
    "Scaler": "scaler",
    "Encoder": "encoder",
    "ML Task": "ml_task",
    "RMSE": "rmse",
    "median Absolute Error": "median_absolute_error",
    "mean Average Precision": "mean_average_precision",
    "macro ROC AUC": "macro_roc_auc",
    "Log Loss": "log_loss",
    "Fit Time": "fit_time",
    "Predict Time": "predict_time",
}


COLUMN_TO_DIMENSION = {v: k for k, v in DIMENSION_TO_COLUMN.items()}


INVERT_COLORMAP = [
    "RMSE",
    "Log Loss",
    "Fit Time",
    "Predict Time",
    "median Absolute Error",
]


class Metadata(pd.DataFrame):
    _metadata = ["project"]

    @staticmethod
    def factory(project, /):
        def dto(summary):
            return dict(
                (
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
        summaries = pd.DataFrame(
            data=pd.DataFrame(
                map(dto, summaries),
                index=pd.MultiIndex.from_arrays(
                    [
                        pd.RangeIndex(len(summaries)),
                        pd.Index(
                            (summary.pop("id") for summary in summaries),
                            name="id",
                            dtype=str,
                        ),
                    ]
                ),
            ),
            copy=False,
        )

        metadata = Metadata(summaries)
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
        """
        Display the interactive plot and controls.

        This method updates the plot widget with the current dataframe state
        before displaying it, ensuring the visualization reflects the most
        recent changes to the data.
        """
        # Recreate the plot widget with the current dataframe state
        self._plot_widget = ModelExplorerWidget(
            dataframe=self,
            dimension_to_column=DIMENSION_TO_COLUMN,
            column_to_dimension=COLUMN_TO_DIMENSION,
            invert_colormap=INVERT_COLORMAP,
        )

        # Display the updated widget
        self._plot_widget.display()
        return ""

    def query_string_selection(self):
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
        >>> query_string = df.query_string_selection()
        >>> filtered_df = df.query(query_string)
        """
        if not hasattr(self, "_plot_widget"):
            return ""

        # First update the selection to ensure we have the latest state
        self._plot_widget.update_selection()

        # Get current task from dropdown
        task = self._plot_widget.task_dropdown.value

        # Build the query string based on current selections and task filter
        query_parts = []

        # Always add task filter
        query_parts.append(f"ml_task == '{task}'")

        # If we have a dataset selected, add that as a filter
        if (
            hasattr(self._plot_widget.dataset_dropdown, "value")
            and self._plot_widget.dataset_dropdown.value
        ):
            dataset_name = self._plot_widget.dataset_dropdown.value
            query_parts.append(f"dataset == '{dataset_name}'")

        # Add selection constraints if any
        for dim_name, range_values in self._plot_widget.current_selection.items():
            # Handle Learner dimension
            if dim_name == "Learner":
                # Find which learner values fall within the selected range
                learner_values = self["learner"].unique()
                learner_codes = pd.Categorical(learner_values).codes

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

            # Handle numerical dimensions
            elif dim_name in COLUMN_TO_DIMENSION.values():
                # Find the column name that corresponds to this dimension label
                col_name = None
                for col, label in COLUMN_TO_DIMENSION.items():
                    if label == dim_name:
                        col_name = col
                        break

                if col_name:
                    min_val, max_val = range_values
                    query_parts.append(
                        f"({col_name} >= {min_val:.6f} and {col_name} <= {max_val:.6f})"
                    )

        # Join all query parts with logical AND
        if query_parts:
            return " and ".join(query_parts)
        else:
            return ""
