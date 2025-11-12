"""Widget for interactive parallel coordinate plots of ML experiment metadata."""

from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML, clear_output, display
from ipywidgets import widgets
from rich.panel import Panel


class Axis(TypedDict):
    """An axis in the parallel coordinate plot."""

    # TODO: Rename to "verbose_name"
    name: str


class MetricAxis(Axis):
    """An axis in the parallel coordinate plot that represents a performance metric."""

    greater_is_better: bool
    type: Literal["time", "regression", "classification"]
    # Whether to show that axis by default
    show: bool


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

    _metrics: dict[str, MetricAxis] = {
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
    _cross_validation_metrics = [
        "fit_time_mean",
        "predict_time_mean",
        "rmse_mean",
        "log_loss_mean",
        "roc_auc_mean",
    ]
    _estimators: dict[str, Axis] = {
        "learner": {"name": "Learner"},
    }
    _dimension_to_column: dict[str, str] = {
        v["name"]: k for k, v in (_metrics | _estimators).items()
    }

    _required_columns: list[str] = (
        ["ml_task", "dataset"]
        + list(_estimators.keys())
        + list(_metrics.keys())
        + _cross_validation_metrics
    )
    _required_index: list[str | None] = [None, "id"]

    def _create_multi_select_dropdown(
        self, options: list[tuple[str, str]], value: list[str], description: str
    ) -> widgets.VBox:
        """Create a compact multi-select dropdown widget.

        This creates a dropdown that shows selected items as a summary text and expands
        to show checkboxes when clicked.

        Parameters
        ----------
        options : list[tuple[str, str]]
            The options to display in the dropdown.
        value : list[str]
            The values that are currently selected.
        description : str
            The description of the dropdown. String shown in the dropdown header.

        Returns
        -------
        widgets.VBox
            The compact multi-select dropdown widget.
        """
        checkboxes = {}
        checkbox_widgets = []

        for label, val in options:
            checkbox = widgets.Checkbox(
                value=val in value,
                description=label,
                indent=False,
                layout=widgets.Layout(width="auto"),
            )
            checkboxes[val] = checkbox
            checkbox_widgets.append(checkbox)

        checkbox_container = widgets.VBox(
            checkbox_widgets, layout=widgets.Layout(max_height="auto")
        )

        accordion = widgets.Accordion(
            children=[checkbox_container],
            layout=widgets.Layout(width="auto"),
        )
        accordion.set_title(0, description.rstrip(":"))
        accordion.selected_index = None
        accordion.add_class("no-padding-accordion")

        widget_container = widgets.VBox([accordion], layout=widgets.Layout(flex="1"))
        widget_container._checkboxes = checkboxes
        widget_container._get_selected_values = lambda: [
            val for val, cb in checkboxes.items() if cb.value
        ]

        return widget_container

    def _check_dataframe_schema(self, dataframe: pd.DataFrame) -> None:
        """Check if the dataframe has the required columns and index.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to check.
        """
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

    def _filter_dataframe(self, ml_task: str, report_type: str) -> pd.DataFrame:
        """Filter report data based on selected ML task and report type.

        Parameters
        ----------
        ml_task : str
            The ML task to filter by.
        report_type : str
            The report type to filter by.

        Returns
        -------
        pd.DataFrame
            The filtered dataframe.
        """
        df = self.dataframe.copy()
        df = df.query(
            f"ml_task.str.contains('{ml_task}') & report_type == '{report_type}'"
        )

        mean_metrics = [col for col in df.columns if col.endswith("_mean")]
        if report_type == "estimator":
            df = df.drop(columns=mean_metrics)
        elif report_type == "cross-validation":
            cols_to_remove = [col.removesuffix("_mean") for col in mean_metrics]
            df = df.drop(columns=cols_to_remove)
            df.columns = [col.removesuffix("_mean") for col in df.columns]
        return df

    def _get_datasets(self, ml_task: str, report_type: str) -> list[str]:
        """Get the unique datasets from the filtered dataframe.

        Parameters
        ----------
        ml_task : str
            The ML task to filter by.
        report_type : str
            The report type to filter by.

        Returns
        -------
        list[str]
            The unique datasets.
        """
        return self._filter_dataframe(ml_task, report_type)["dataset"].unique()

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

        # Figure out dropdown defaults
        for ml_task, report_type in [
            ("classification", "estimator"),
            ("regression", "estimator"),
            ("classification", "cross-validation"),
            ("regression", "cross-validation"),
        ]:
            if not self._filter_dataframe(ml_task, report_type).empty:
                default_task = ml_task
                default_report_type = report_type
                break

        self._report_type_dropdown = widgets.Dropdown(
            options=[
                ("Estimator", "estimator"),
                ("Cross-validation", "cross-validation"),
            ],
            value=default_report_type,
            description="Report Type:",
            disabled=False,
            layout=widgets.Layout(flex="1"),
        )

        self._task_dropdown = widgets.Dropdown(
            options=[
                ("Classification", "classification"),
                ("Regression", "regression"),
            ],
            value=default_task,
            description="Task Type:",
            disabled=False,
            layout=widgets.Layout(flex="1"),
        )

        default_dataset = self._get_datasets(default_task, default_report_type)
        self._dataset_dropdown = widgets.Dropdown(
            options=default_dataset,
            description="Dataset:",
            disabled=False,
            layout=widgets.Layout(flex="1"),
        )

        self._computation_metrics_dropdown: dict[str, widgets.SelectMultiple] = {}
        self._statistical_metrics_dropdown: dict[str, widgets.SelectMultiple] = {}

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

        self._computation_metrics_dropdown = {}
        self._statistical_metrics_dropdown = {}

        computation_metrics = ["fit_time", "predict_time"]
        computation_options = [
            (self._metrics[metric]["name"], metric) for metric in computation_metrics
        ]
        for task in ["classification", "regression"]:
            self._computation_metrics_dropdown[task] = (
                self._create_multi_select_dropdown(
                    options=computation_options,
                    value=[],
                    description="Computation Metrics:",
                )
            )

        classification_statistical = ["roc_auc", "log_loss"]
        regression_statistical = ["rmse"]

        classification_stat_options = [
            (self._metrics[metric]["name"], metric)
            for metric in classification_statistical
        ]
        regression_stat_options = [
            (self._metrics[metric]["name"], metric) for metric in regression_statistical
        ]

        classification_stat_default = [
            metric
            for metric in classification_statistical
            if self._metrics[metric]["show"]
        ]
        regression_stat_default = [
            metric for metric in regression_statistical if self._metrics[metric]["show"]
        ]

        self._statistical_metrics_dropdown["classification"] = (
            self._create_multi_select_dropdown(
                options=classification_stat_options,
                value=classification_stat_default,
                description="Statistical Metrics:",
            )
        )

        self._statistical_metrics_dropdown["regression"] = (
            self._create_multi_select_dropdown(
                options=regression_stat_options,
                value=regression_stat_default,
                description="Statistical Metrics:",
            )
        )
        self._color_metric_dropdown: dict[str, widgets.Dropdown] = {
            "classification": widgets.Dropdown(
                options=[
                    self._metrics[metric]["name"]
                    for metric in metrics_for_classification
                ],
                value="Log Loss",
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(flex="1"),
            ),
            "regression": widgets.Dropdown(
                options=[
                    self._metrics[metric]["name"] for metric in metrics_for_regression
                ],
                value="RMSE",
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(flex="1"),
            ),
        }
        controls_header = widgets.HBox(
            [
                self._report_type_dropdown,
                self._task_dropdown,
                self._dataset_dropdown,
            ],
            layout=widgets.Layout(width="100%"),
        )

        self.classification_metrics_box = widgets.HBox(
            [
                self._computation_metrics_dropdown["classification"],
                self._statistical_metrics_dropdown["classification"],
                self._color_metric_dropdown["classification"],
            ],
            layout=widgets.Layout(width="100%"),
        )

        self.regression_metrics_box = widgets.HBox(
            [
                self._computation_metrics_dropdown["regression"],
                self._statistical_metrics_dropdown["regression"],
                self._color_metric_dropdown["regression"],
            ],
            layout=widgets.Layout(width="100%"),
        )
        controls_metrics = widgets.VBox(
            [self.classification_metrics_box, self.regression_metrics_box],
            layout=widgets.Layout(width="100%"),
        )
        controls = widgets.VBox([controls_header, controls_metrics])

        # callbacks
        self._report_type_dropdown.observe(self._on_report_type_change, names="value")
        self._task_dropdown.observe(self._on_task_change, names="value")
        self._dataset_dropdown.observe(self._update_plot, names="value")
        for task in ["classification", "regression"]:
            # Add observers to all checkboxes in the multi-select dropdowns
            for checkbox in self._computation_metrics_dropdown[
                task
            ]._checkboxes.values():
                checkbox.observe(self._update_plot, names="value")
            for checkbox in self._statistical_metrics_dropdown[
                task
            ]._checkboxes.values():
                checkbox.observe(self._update_plot, names="value")
            self._color_metric_dropdown[task].observe(self._update_plot, names="value")

        self.output = widgets.Output(layout=widgets.Layout(width="100%"))

        self._update_task_widgets(ml_task=self._task_dropdown.value)
        self._layout = widgets.VBox(
            [controls, self.output],
            layout=widgets.Layout(width="100%", overflow="hidden"),
        )

    def _update_dataset_dropdown(self, datasets: list[str]) -> None:
        """Update the dataset dropdown options.

        Parameters
        ----------
        datasets : list[str]
            The datasets to display in the dropdown.
        """
        self._dataset_dropdown.options = datasets
        if len(datasets):
            self._dataset_dropdown.value = datasets[0]

    def _update_task_widgets(self, ml_task: str) -> None:
        """Update widgets that are dependent on the selected task.

        Parameters
        ----------
        ml_task : str
            The task to display in the dropdown.
        """
        if ml_task == "classification":
            self.classification_metrics_box.layout.display = ""
            self._color_metric_dropdown["classification"].layout.display = None
            self.regression_metrics_box.layout.display = "none"
            self._color_metric_dropdown["regression"].layout.display = "none"
        else:  # ml_task == "regression"
            self.classification_metrics_box.layout.display = "none"
            self._color_metric_dropdown["classification"].layout.display = "none"
            self.regression_metrics_box.layout.display = ""
            self._color_metric_dropdown["regression"].layout.display = None

    def _on_report_type_change(self, change: dict[str, Any]) -> None:
        """Handle report type dropdown change events.

        Updates the dataset dropdown options based on the selected report type
        and refreshes the widget visibility and plot.

        Parameters
        ----------
        change : dict[str, Any]
            dictionary containing information about the widget change,
            including the new value under the 'new' key.
        """
        ml_task, report_type = self._task_dropdown.value, change["new"]
        self._update_dataset_dropdown(
            self._get_datasets(ml_task=ml_task, report_type=report_type)
        )
        self._update_task_widgets(ml_task=ml_task)
        self._update_plot()
        self.update_selection()

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
        ml_task, report_type = change["new"], self._report_type_dropdown.value
        self._update_dataset_dropdown(
            self._get_datasets(ml_task=ml_task, report_type=report_type)
        )
        self._update_task_widgets(ml_task=ml_task)
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

            ml_task = self._task_dropdown.value
            report_type = self._report_type_dropdown.value

            if (
                not hasattr(self._dataset_dropdown, "value")
                or not self._dataset_dropdown.value
            ):
                display(widgets.HTML("No dataset available for selected task."))
                return

            df_dataset = self._filter_dataframe(ml_task, report_type).query(
                "dataset == @self._dataset_dropdown.value"
            )
            for col in df_dataset.select_dtypes(include=["category"]).columns:
                df_dataset[col] = df_dataset[col].cat.remove_unused_categories()

            # Get selected metrics from both dropdowns
            selected_computation_metrics = self._computation_metrics_dropdown[
                ml_task
            ]._get_selected_values()
            selected_statistical_metrics = self._statistical_metrics_dropdown[
                ml_task
            ]._get_selected_values()
            selected_metrics = (
                selected_computation_metrics + selected_statistical_metrics
            )
            color_metric = self._dimension_to_column[
                self._color_metric_dropdown[ml_task].value
            ]

            dimensions = [
                {
                    "label": "Learner",
                    "values": self._add_jitter_to_categorical(
                        self.seed, df_dataset["learner"]
                    ),
                    "ticktext": df_dataset["learner"].cat.categories,
                    "tickvals": np.arange(len(df_dataset["learner"].cat.categories)),
                }
            ]

            dimensions.extend(
                {
                    "label": self._metrics[col]["name"],
                    # convert to float in case that the column has None values and
                    # thus is object type
                    "values": df_dataset[col].astype(float).fillna(0),
                }
                for col in selected_metrics  # use the order defined in the constructor
            )

            colorscale = (
                "Viridis"
                if self._metrics[color_metric]["greater_is_better"]
                else "Viridis_r"
            )
            fig = go.FigureWidget(
                data=go.Parcoords(
                    line={
                        "color": df_dataset[color_metric].fillna(0),
                        "colorscale": colorscale,
                        "showscale": True,
                        "colorbar": {"title": self._metrics[color_metric]["name"]},
                    },
                    dimensions=dimensions,
                    labelangle=-30,
                )
            )

            fig.update_layout(
                font={"size": 18},
                height=500,
                margin={"l": 250, "r": 0, "t": 120, "b": 30},
            )

            fig.data[0].on_selection(self.update_selection)  # callback

            self.current_fig = fig
            # It is important to set autosize after the figure is displayed so that the
            # width matches the parent container. However, it is not responsive to width
            # resizing, but this is the only way to achieve the correct width. This
            # issue can be tracked in the following bug report:
            # https://github.com/plotly/plotly.py/issues/5208
            display(fig)
            fig.layout.autosize = True

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

        display(
            HTML(
                """
<style>
    /* Text-based widgets */
    .widget-text input,
    .widget-textarea textarea,
    .widget-password input {
        font-size: 16px !important;
    }

    /* Labels and descriptions */
    .widget-label,
    .widget-label-basic {
        font-size: 16px !important;
        min-width: fit-content !important;
    }

    /* Buttons */
    .widget-button,
    .widget-toggle-button {
        font-size: 16px !important;
    }

    /* Dropdowns and select widgets */
    .widget-dropdown select,
    .widget-select select {
        font-size: 16px !important;
    }

    /* Slider readouts */
    .widget-readout {
        font-size: 16px !important;
    }

    /* HTML widgets */
    .widget-html,
    .widget-html-content {
        font-size: 16px !important;
    }

    /* Custom SelectMultiple Dropdown */
    .jupyter-widget-Collapse-header {
        font-size: 16px !important;
        font-weight: normal !important;
    }
</style>
"""
            )
        )

        display(self._layout)
        self._update_plot()
        self.update_selection()
