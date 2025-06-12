"""Widget for interactive parallel coordinate plots of ML experiment metadata."""

from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output, display
from ipywidgets import widgets
from rich.panel import Panel


class ModelExplorerWidget:
    """
    Widget for interactive parallel coordinate plots of ML experiment metadata.

    This class creates and manages interactive widgets and parallel coordinate plots
    for visually exploring machine learning experiment results. It allows users to
    filter and compare experiments across different metrics, learners, and datasets.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the experiment metadata with columns for metrics,
        learners, datasets, and ML task types.
    seed : int, default=0
        Random seed for jittering categorical columns to improve visualization.

    Attributes
    ----------
    current_fig : go.FigureWidget or None
        The currently displayed plotly figure.
    current_selection : dict
        Dictionary containing the current user selection criteria.
    """

    _plot_width: int = 800
    _metrics: dict[str, dict[str, str | bool]] = {
        "fit_time": {
            "name": "Fit Time",
            "greater_is_better": False,
            "type": "time",
            "show": False,
        },
        "predict_time": {
            "name": "Predict Time",
            "greater_is_better": False,
            "type": "time",
            "show": False,
        },
        "rmse": {
            "name": "RMSE",
            "greater_is_better": False,
            "type": "regression",
            "show": True,
        },
        "log_loss": {
            "name": "Log Loss",
            "greater_is_better": False,
            "type": "classification",
            "show": True,
        },
        "roc_auc": {
            "name": "Macro ROC AUC",
            "greater_is_better": True,
            "type": "classification",
            "show": True,
        },
    }
    _estimators: dict[str, dict[str, Any]] = {
        "learner": {"name": "Learner"},
    }
    _dimension_to_column: dict[str, str] = {
        cast(str, v["name"]): k for k, v in {**_metrics, **_estimators}.items()
    }

    _required_columns: list[str] = (
        ["ml_task", "dataset"] + list(_estimators.keys()) + list(_metrics.keys())
    )
    _required_index: list[str | None] = [None, "id"]

    def _check_dataframe_schema(self, dataframe: pd.DataFrame) -> None:
        """Check if the dataframe has the required columns and index."""
        if not all(col in dataframe.columns for col in self._required_columns):
            raise ValueError(
                f"Dataframe is missing required columns: {self._required_columns}"
            )
        if not all(col in dataframe.index.names for col in self._required_index):
            raise ValueError(
                f"Dataframe is missing required index: {self._required_index}"
            )
        if dataframe["learner"].dtype != "category":
            raise ValueError("Learner column must be a categorical column")

    def __init__(self, dataframe: pd.DataFrame, seed: int = 0) -> None:
        if dataframe.empty:
            self.dataframe = dataframe
            self.seed = seed
            return None

        self._check_dataframe_schema(dataframe)
        self.dataframe = dataframe
        self.seed = seed

        self.current_fig: go.FigureWidget | None = None
        self.current_selection: dict[str, Any] = {}

        self._clf_datasets: np.ndarray = self.dataframe.query(
            "ml_task.str.contains('classification')"
        )["dataset"].unique()
        self._reg_datasets: np.ndarray = self.dataframe.query(
            "ml_task.str.contains('regression')"
        )["dataset"].unique()

        default_task = "classification" if len(self._clf_datasets) else "regression"
        self._task_dropdown = widgets.Dropdown(
            options=[
                ("Classification", "classification"),
                ("Regression", "regression"),
            ],
            value=default_task,
            description="Task Type:",
            disabled=False,
            layout=widgets.Layout(width="200px"),
        )

        default_dataset = (
            self._clf_datasets
            if default_task == "classification"
            else self._reg_datasets
        )
        self._dataset_dropdown = widgets.Dropdown(
            options=default_dataset,
            description="Dataset:",
            disabled=False,
            layout=widgets.Layout(width="250px"),
        )

        self._metric_checkboxes: dict[str, dict[str, widgets.Checkbox]] = {
            "classification": {},
            "regression": {},
        }
        for metric in self._metrics:
            default_value = self._metrics[metric]["show"]
            metric_type = cast(str, self._metrics[metric]["type"])
            if metric_type == "time":
                # the "time" metrics should be added to all the different types
                # (i.e. classification and regression)
                for metric_type in self._metric_checkboxes:
                    self._metric_checkboxes[metric_type][metric] = widgets.Checkbox(
                        indent=False,
                        value=default_value,
                        description=cast(str, self._metrics[metric]["name"]),
                        disabled=False,
                        layout=widgets.Layout(width="auto", margin="0px 10px 0px 0px"),
                    )
            else:
                self._metric_checkboxes[metric_type][metric] = widgets.Checkbox(
                    indent=False,
                    value=default_value,
                    description=cast(str, self._metrics[metric]["name"]),
                    disabled=False,
                    layout=widgets.Layout(width="auto", margin="0px 10px 0px 0px"),
                )

        metrics_for_classification = [
            metric
            for metric in self._metrics
            if self._metrics[metric]["type"] in ("classification", "time")
        ]
        metrics_for_regression = [
            metric
            for metric in self._metrics
            if self._metrics[metric]["type"] in ("regression", "time")
        ]
        self._color_metric_dropdown: dict[str, widgets.Dropdown] = {
            "classification": widgets.Dropdown(
                options=[
                    cast(str, self._metrics[metric]["name"])
                    for metric in metrics_for_classification
                ],
                value="Log Loss",
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(width="200px"),
            ),
            "regression": widgets.Dropdown(
                options=[
                    cast(str, self._metrics[metric]["name"])
                    for metric in metrics_for_regression
                ],
                value="RMSE",
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(width="200px"),
            ),
        }
        self.classification_metrics_box = widgets.HBox(
            [
                self._metric_checkboxes["classification"][metric]
                for metric in metrics_for_classification
            ]
        )
        self.regression_metrics_box = widgets.HBox(
            [
                self._metric_checkboxes["regression"][metric]
                for metric in metrics_for_regression
            ]
        )

        controls_header = widgets.GridBox(
            [
                self._task_dropdown,
                self._dataset_dropdown,
                self._color_metric_dropdown["classification"],
                self._color_metric_dropdown["regression"],
            ],
            layout=widgets.Layout(
                width=f"{self._plot_width}px",
                grid_template_columns="repeat(4, auto)",
                grid_gap="5px",
                align_items="center",
            ),
        )

        clf_computation_row = widgets.GridBox(
            [
                widgets.Label(
                    value="Computation Metrics: ",
                    layout=widgets.Layout(padding="5px 0px"),
                ),
                self._metric_checkboxes["classification"]["fit_time"],
                self._metric_checkboxes["classification"]["predict_time"],
            ],
            layout=widgets.Layout(
                width=f"{self._plot_width}px",
                grid_template_columns="200px auto auto",
                align_items="center",
            ),
        )
        clf_statistical_row = widgets.GridBox(
            [
                widgets.Label(
                    value="Statistical Metrics: ",
                    layout=widgets.Layout(padding="5px 0px"),
                ),
                self._metric_checkboxes["classification"]["roc_auc"],
                self._metric_checkboxes["classification"]["log_loss"],
            ],
            layout=widgets.Layout(
                width=f"{self._plot_width}px",
                grid_template_columns="200px auto auto auto",
                align_items="center",
            ),
        )
        self.classification_metrics_box = widgets.GridBox(
            [clf_computation_row, clf_statistical_row],
            layout=widgets.Layout(
                width=f"{self._plot_width}px",
                grid_template_rows="auto auto",
                grid_gap="5px",
                align_items="center",
            ),
        )
        reg_computation_row = widgets.GridBox(
            [
                widgets.Label(
                    value="Computation Metrics: ",
                    layout=widgets.Layout(padding="5px 0px"),
                ),
                self._metric_checkboxes["regression"]["fit_time"],
                self._metric_checkboxes["regression"]["predict_time"],
            ],
            layout=widgets.Layout(
                width=f"{self._plot_width}px",
                grid_template_columns="200px auto auto",
                align_items="center",
            ),
        )
        reg_statistical_row = widgets.GridBox(
            [
                widgets.Label(
                    value="Statistical Metrics: ",
                    layout=widgets.Layout(padding="5px 0px"),
                ),
                self._metric_checkboxes["regression"]["rmse"],
            ],
            layout=widgets.Layout(
                width=f"{self._plot_width}px",
                grid_template_columns="200px auto auto",
                align_items="center",
            ),
        )
        self.regression_metrics_box = widgets.GridBox(
            [reg_computation_row, reg_statistical_row],
            layout=widgets.Layout(
                width=f"{self._plot_width}px",
                grid_template_rows="auto auto",
                grid_gap="5px",
                align_items="center",
            ),
        )
        controls_metrics = widgets.GridBox(
            [self.classification_metrics_box, self.regression_metrics_box],
            layout=widgets.Layout(
                width=f"{self._plot_width}px",
                grid_template_rows="auto auto",
                grid_gap="10px",
                align_items="center",
            ),
        )
        controls = widgets.GridBox(
            [controls_header, controls_metrics],
            layout=widgets.Layout(
                width=f"{self._plot_width}px",
                grid_template_rows="auto auto",
                grid_gap="10px",
                align_items="center",
                margin="0px 0px 5px 0px",
            ),
        )

        # callbacks
        self._task_dropdown.observe(self._on_task_change, names="value")
        self._dataset_dropdown.observe(self._update_plot, names="value")
        for task in self._metric_checkboxes:
            for metric in self._metric_checkboxes[task]:
                self._metric_checkboxes[task][metric].observe(
                    self._update_plot, names="value"
                )
            self._color_metric_dropdown[task].observe(self._update_plot, names="value")

        self.output = widgets.Output(
            layout=widgets.Layout(width=f"{self._plot_width}px", margin="0px")
        )

        self._update_task_widgets()
        self._layout = widgets.VBox(
            [controls, self.output],
            layout=widgets.Layout(width=f"{self._plot_width}px"),
        )

    def _update_task_widgets(self) -> None:
        """Update widget visibility based on the currently selected task."""
        task = self._task_dropdown.value

        if task == "classification":
            self.classification_metrics_box.layout.display = ""
            self._color_metric_dropdown["classification"].layout.display = ""

            self.regression_metrics_box.layout.display = "none"
            self._color_metric_dropdown["regression"].layout.display = "none"
        else:
            self.classification_metrics_box.layout.display = "none"
            self._color_metric_dropdown["classification"].layout.display = "none"

            self.regression_metrics_box.layout.display = ""
            self._color_metric_dropdown["regression"].layout.display = ""

    def _on_task_change(self, change: dict[str, Any]) -> None:
        """Handle task dropdown change events.

        Updates the dataset dropdown options based on the selected task
        and refreshes the widget visibility and plot.

        Parameters
        ----------
        change : dict[str, Any]
            dictionary containing information about the widget change,
            including the new value under the 'new' key.
        """
        task = change["new"]

        if task == "classification":
            self._dataset_dropdown.options = self._clf_datasets
            if len(self._clf_datasets):
                self._dataset_dropdown.value = self._clf_datasets[0]
        else:
            self._dataset_dropdown.options = self._reg_datasets
            if len(self._reg_datasets):
                self._dataset_dropdown.value = self._reg_datasets[0]

        self._update_task_widgets()
        self._update_plot()
        self.update_selection()

    @staticmethod
    def _add_jitter_to_categorical(
        seed: int, categorical_series: pd.Series, amount: float = 0.01
    ) -> np.ndarray:
        """Add jitter to categorical values to improve visualization in parallel plots.

        Jitter is not applied when there is a single category.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility.
        categorical_series : pd.Series
            Categorical series to jitter.
        amount : float, default=0.01
            Amount of jitter to add.

        Returns
        -------
        np.ndarray
            Array of encoded categorical values with jitter applied.
        """
        if categorical_series.cat.categories.size == 1:
            return categorical_series.cat.codes.to_numpy()

        rng = np.random.default_rng(seed)
        encoded_categories = categorical_series.cat.codes.to_numpy()
        jitter = rng.uniform(-amount, amount, size=len(encoded_categories))
        for sign, cat in zip([1, -1], [0, len(encoded_categories) - 1], strict=False):
            jitter[encoded_categories == cat] = (
                np.abs(
                    jitter[encoded_categories == cat],
                    out=jitter[encoded_categories == cat],
                )
                * sign
            )

        return encoded_categories + jitter

    def _update_plot(self, change: dict[str, Any] | None = None) -> None:
        """Update the parallel coordinates plot based on the selected options.

        Creates a new plotly figure with dimensions for the selected metrics
        and displays it in the output area.

        Parameters
        ----------
        change : dict, default=None
            dictionary containing information about a widget change event.
            Not used directly but required for widget callback compatibility.
        """
        with self.output:
            clear_output(wait=True)

            task = self._task_dropdown.value

            if (
                not hasattr(self._dataset_dropdown, "value")
                or not self._dataset_dropdown.value
            ):
                display(widgets.HTML("No dataset available for selected task."))
                return

            df_dataset = self.dataframe.query(
                "dataset == @self._dataset_dropdown.value"
            ).copy()
            for col in df_dataset.select_dtypes(include=["category"]).columns:
                df_dataset[col] = df_dataset[col].cat.remove_unused_categories()

            selected_metrics = [
                metric
                for metric in self._metric_checkboxes[task]
                if self._metric_checkboxes[task][metric].value
            ]
            color_metric = self._dimension_to_column[
                self._color_metric_dropdown[task].value
            ]

            dimensions = []
            dimensions.append(
                dict(
                    label="Learner",
                    values=self._add_jitter_to_categorical(
                        self.seed, df_dataset["learner"]
                    ),
                    ticktext=df_dataset["learner"].cat.categories,
                    tickvals=np.arange(len(df_dataset["learner"].cat.categories)),
                )
            )

            for col in selected_metrics:  # use the order defined in the constructor
                dimensions.append(
                    dict(
                        label=cast(str, self._metrics[col]["name"]),
                        values=df_dataset[col].fillna(0),
                    )
                )

            colorscale = (
                "Viridis"
                if cast(bool, self._metrics[color_metric]["greater_is_better"])
                else "Viridis_r"
            )
            fig = go.FigureWidget(
                data=go.Parcoords(
                    line=dict(
                        color=df_dataset[color_metric].fillna(0),
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(
                            title=cast(str, self._metrics[color_metric]["name"])
                        ),
                    ),
                    dimensions=dimensions,
                    labelangle=-30,
                )
            )

            fig.update_layout(
                font=dict(size=16),
                height=500,
                width=self._plot_width,
                margin=dict(l=250, r=150, t=120, b=30),
                autosize=False,
            )

            fig.data[0].on_selection(self.update_selection)  # callback

            self.current_fig = fig
            display(fig)

    def update_selection(
        self, trace=None, points=None, selector=None
    ) -> "ModelExplorerWidget":
        """Update the current_selection attribute with the current filter state.

        Parameters
        ----------
        trace : trace, default=None
            The trace that triggered the selection change.
        points : list, default=None
            The points affected by the selection change.
        selector : dict, default=None
            The selector that triggered the selection change.

        Returns
        -------
        ModelExplorerWidget
            Self for method chaining.
        """
        selection_data = {
            "ml_task": self._task_dropdown.value,
            "dataset": self._dataset_dropdown.value,
        }

        if self.current_fig is not None:
            selection_data.update(
                {
                    self._dimension_to_column[dim.label]: dim.constraintrange
                    for dim in self.current_fig.data[0].dimensions
                    if hasattr(dim, "constraintrange") and dim.constraintrange
                }
            )

        self.current_selection = selection_data

        return self

    def display(self) -> None:
        """Display the widget interface and initialize the plot."""
        if self.dataframe.empty:
            from skore import console  # avoid circular import

            content = (
                "No report found in the project. Use the `put` method to add reports."
            )
            console.print(
                Panel(
                    content,
                    title="[bold cyan]Empty Project Metadata[/bold cyan]",
                    expand=False,
                    border_style="orange1",
                )
            )
            return None

        display(self._layout)
        self._update_plot()
        self.update_selection()
