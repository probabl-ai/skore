import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output, display
from ipywidgets import widgets
from plotly.graph_objects import FigureWidget


class ModelExplorerWidget:
    """
    Widget for interactive parallel coordinate plots of ML experiment metadata.

    This class handles the creation and management of interactive widgets
    and the parallel coordinate plot for exploring ML experiment data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the experiment metadata.
    dimension_to_column : dict
        Mapping from display names to dataframe column names.
    column_to_dimension : dict
        Mapping from dataframe column names to display names.
    invert_colormap : list
        List of metrics where lower values are better (for inverting colormap).
    clf_datasets : list
        List of classification dataset names.
    reg_datasets : list
        List of regression dataset names.
    seed : int, default=0
        Seed for the jittering categorical columns.
    """

    _plot_width = 800

    def __init__(
        self,
        dataframe,
        dimension_to_column,
        column_to_dimension,
        invert_colormap,
        seed=0,
    ):
        self.df = dataframe
        self.dimension_to_column = dimension_to_column
        self.column_to_dimension = column_to_dimension
        self.invert_colormap = invert_colormap
        self.seed = seed

        self.current_fig = None
        self.current_dimensions = None
        self.current_selection = {}

        classification_metrics = ["mean Average Precision", "macro ROC AUC", "Log Loss"]
        regression_metrics = ["median Absolute Error", "RMSE"]
        time_metrics = ["Fit Time", "Predict Time"]

        self._clf_datasets = self.df.query("ml_task == 'classification'")[
            "dataset"
        ].unique()
        self._reg_datasets = self.df.query("ml_task == 'regression'")[
            "dataset"
        ].unique()

        self.task_dropdown = widgets.Dropdown(
            options=[
                ("Classification", "classification"),
                ("Regression", "regression"),
            ],
            value="classification",
            description="Task Type:",
            disabled=False,
            layout=widgets.Layout(width="200px"),
        )

        self.dataset_dropdown = widgets.Dropdown(
            options=self._clf_datasets,
            description="Dataset:",
            disabled=False,
            layout=widgets.Layout(width="250px"),
        )

        self.metric_checkboxes = {"classification": {}, "regression": {}}
        for metric in time_metrics + classification_metrics:
            default_value = metric not in time_metrics
            self.metric_checkboxes["classification"][metric] = widgets.Checkbox(
                indent=False,
                value=default_value,
                description=metric,
                disabled=False,
                layout=widgets.Layout(width="auto", margin="0px 10px 0px 0px"),
            )
        for metric in time_metrics + regression_metrics:
            default_value = metric not in time_metrics
            self.metric_checkboxes["regression"][metric] = widgets.Checkbox(
                indent=False,
                value=default_value,
                description=metric,
                disabled=False,
                layout=widgets.Layout(width="auto", margin="0px 10px 0px 0px"),
            )
        self.color_metric_dropdown = {
            "classification": widgets.Dropdown(
                options=time_metrics + classification_metrics,
                value="Log Loss",
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(width="200px"),
            ),
            "regression": widgets.Dropdown(
                options=time_metrics + regression_metrics,
                value="RMSE",
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(width="200px"),
            ),
        }
        self.classification_metrics_box = widgets.HBox(
            [
                self.metric_checkboxes["classification"][metric]
                for metric in time_metrics + classification_metrics
            ]
        )
        self.regression_metrics_box = widgets.HBox(
            [
                self.metric_checkboxes["regression"][metric]
                for metric in time_metrics + regression_metrics
            ]
        )

        # callbacks
        self.task_dropdown.observe(self._on_task_change, names="value")
        self.dataset_dropdown.observe(self._update_plot, names="value")
        for task in self.metric_checkboxes:
            for metric in self.metric_checkboxes[task]:
                self.metric_checkboxes[task][metric].observe(
                    self._update_plot, names="value"
                )
            self.color_metric_dropdown[task].observe(self._update_plot, names="value")

        self.output = widgets.Output()

    def _on_task_change(self, change):
        """Handle task dropdown change event."""
        task = change["new"]

        if task == "classification":
            self.dataset_dropdown.options = self._clf_datasets
            if len(self._clf_datasets):
                self.dataset_dropdown.value = self._clf_datasets[0]
        else:
            self.dataset_dropdown.options = self._reg_datasets
            if len(self._reg_datasets):
                self.dataset_dropdown.value = self._reg_datasets[0]

        self._update_task_widgets()
        self._update_plot()

    def _update_task_widgets(self):
        """Update widget visibility based on selected task."""
        task = self.task_dropdown.value

        if task == "classification":
            self.classification_metrics_box.layout.display = ""
            self.color_metric_dropdown["classification"].layout.display = ""

            self.regression_metrics_box.layout.display = "none"
            self.color_metric_dropdown["regression"].layout.display = "none"
        else:
            self.classification_metrics_box.layout.display = "none"
            self.color_metric_dropdown["classification"].layout.display = "none"

            self.regression_metrics_box.layout.display = ""
            self.color_metric_dropdown["regression"].layout.display = ""

    def _add_jitter_to_categorical(self, categorical_series, amount=0.01):
        """Add jitter to categorical values to improve visualization."""
        rng = np.random.default_rng(self.seed)
        encoded_categories = categorical_series.cat.codes
        jitter = rng.uniform(-amount, amount, size=len(encoded_categories))
        for sign, cat in zip([1, -1], [0, len(encoded_categories) - 1]):
            jitter[encoded_categories == cat] = (
                np.abs(
                    jitter[encoded_categories == cat],
                    out=jitter[encoded_categories == cat],
                )
                * sign
            )

        return encoded_categories + jitter

    def _update_plot(self, change=None):
        """Update the parallel coordinates plot based on the selected options."""
        with self.output:
            clear_output(wait=True)

            task = self.task_dropdown.value

            if (
                not hasattr(self.dataset_dropdown, "value")
                or not self.dataset_dropdown.value
            ):
                display(widgets.HTML("No dataset available for selected task."))
                return

            df_dataset = self.df.query("dataset == @self.dataset_dropdown.value").copy()
            for col in df_dataset.select_dtypes(include=["category"]).columns:
                df_dataset[col] = df_dataset[col].cat.remove_unused_categories()

            selected_metrics = [
                metric
                for metric in self.metric_checkboxes[task]
                if self.metric_checkboxes[task][metric].value
            ]
            color_metric = self.color_metric_dropdown[task].value

            selected_columns = [
                self.dimension_to_column[metric] for metric in selected_metrics
            ]
            color_column = self.dimension_to_column[color_metric]

            dimensions = []
            dimensions.append(
                dict(
                    label="Learner",
                    values=self._add_jitter_to_categorical(df_dataset["learner"]),
                    ticktext=df_dataset["learner"].cat.categories,
                    tickvals=np.arange(len(df_dataset["learner"].cat.categories)),
                )
            )

            for col in selected_columns:  # use the order defined in the constructor
                dimensions.append(
                    dict(
                        label=self.column_to_dimension[col],
                        values=df_dataset[col].fillna(0),
                    )
                )

            fig = go.Figure(
                data=go.Parcoords(
                    line=dict(
                        color=df_dataset[color_column].fillna(0),
                        colorscale=(
                            "Viridis_r"
                            if color_metric in self.invert_colormap
                            else "Viridis"
                        ),
                        showscale=True,
                        colorbar=dict(title=color_metric),
                    ),
                    dimensions=dimensions,
                    labelangle=-30,
                )
            )

            fig.update_layout(
                font=dict(size=16),
                height=500,
                width=self._plot_width,
                margin=dict(l=200, r=150, t=120, b=30),
            )

            f_widget = FigureWidget(fig)

            # Setup callback for selection changes
            def selection_change_callback(trace, points, selector):
                self.update_selection()

            # Add callback to detect selection changes
            f_widget.data[0].on_selection(selection_change_callback)

            # Store current figure and dimensions
            self.current_fig = f_widget
            self.current_dimensions = dimensions

            display(f_widget)

    def update_selection(self):
        """
        Update the selection based on the current state of the plot.
        """
        if not self.current_fig or not hasattr(self.current_fig.data[0], "dimensions"):
            return self

        # Extract the constraint ranges from the plot data
        selection_data = {}
        for i, dim in enumerate(self.current_fig.data[0].dimensions):
            if hasattr(dim, "constraintrange") and dim.constraintrange:
                dim_name = dim.label
                selection_data[dim_name] = dim.constraintrange

        # Store the selection data
        self.current_selection = selection_data
        return self

    def create_layout(self):
        """Create and return the widget layout."""
        # Set consistent width for controls and plot
        controls_width = "800px"

        # Create controls layout with specific spacing using GridBox
        controls_header = widgets.GridBox(
            [
                self.task_dropdown,
                self.dataset_dropdown,
                self.color_metric_dropdown["classification"],
                self.color_metric_dropdown["regression"],
            ],
            layout=widgets.Layout(
                width=controls_width,
                grid_template_columns="repeat(4, auto)",
                grid_gap="5px",
                align_items="center",
            ),
        )

        # Labels for the different metric types
        comp_metrics_label_clf = widgets.Label(
            value="Computation Metrics: ", layout=widgets.Layout(padding="5px 0px")
        )

        stat_metrics_label_clf = widgets.Label(
            value="Statistical Metrics: ", layout=widgets.Layout(padding="5px 0px")
        )

        comp_metrics_label_reg = widgets.Label(
            value="Computation Metrics: ", layout=widgets.Layout(padding="5px 0px")
        )

        stat_metrics_label_reg = widgets.Label(
            value="Statistical Metrics: ", layout=widgets.Layout(padding="5px 0px")
        )

        # Classification metrics - Using GridBox for better alignment
        # First row: Computation metrics
        clf_computation_row = widgets.GridBox(
            [
                comp_metrics_label_clf,
                self.metric_checkboxes["classification"]["Fit Time"],
                self.metric_checkboxes["classification"]["Predict Time"],
            ],
            layout=widgets.Layout(
                width=controls_width,
                grid_template_columns="200px auto auto",
                align_items="center",
            ),
        )

        # Second row: Statistical metrics
        clf_statistical_row = widgets.GridBox(
            [
                stat_metrics_label_clf,
                self.metric_checkboxes["classification"]["mean Average Precision"],
                self.metric_checkboxes["classification"]["macro ROC AUC"],
                self.metric_checkboxes["classification"]["Log Loss"],
            ],
            layout=widgets.Layout(
                width=controls_width,
                grid_template_columns="200px auto auto auto",
                align_items="center",
            ),
        )

        # Combined classification metrics container
        self.classification_metrics_box = widgets.GridBox(
            [clf_computation_row, clf_statistical_row],
            layout=widgets.Layout(
                width=controls_width,
                grid_template_rows="auto auto",
                grid_gap="5px",
                align_items="center",
            ),
        )

        # Regression metrics - Using GridBox for better alignment
        # First row: Computation metrics
        reg_computation_row = widgets.GridBox(
            [
                comp_metrics_label_reg,
                self.metric_checkboxes["regression"]["Fit Time"],
                self.metric_checkboxes["regression"]["Predict Time"],
            ],
            layout=widgets.Layout(
                width=controls_width,
                grid_template_columns="200px auto auto",
                align_items="center",
            ),
        )

        # Second row: Statistical metrics
        reg_statistical_row = widgets.GridBox(
            [
                stat_metrics_label_reg,
                self.metric_checkboxes["regression"]["median Absolute Error"],
                self.metric_checkboxes["regression"]["RMSE"],
            ],
            layout=widgets.Layout(
                width=controls_width,
                grid_template_columns="200px auto auto",
                align_items="center",
            ),
        )

        # Combined regression metrics container
        self.regression_metrics_box = widgets.GridBox(
            [reg_computation_row, reg_statistical_row],
            layout=widgets.Layout(
                width=controls_width,
                grid_template_rows="auto auto",
                grid_gap="5px",
                align_items="center",
            ),
        )

        # Create a container for the metrics
        controls_metrics = widgets.GridBox(
            [self.classification_metrics_box, self.regression_metrics_box],
            layout=widgets.Layout(
                width=controls_width,
                grid_template_rows="auto auto",
                grid_gap="10px",
                align_items="center",
            ),
        )

        # Apply consistent width to entire controls container
        controls = widgets.GridBox(
            [controls_header, controls_metrics],
            layout=widgets.Layout(
                width=controls_width,
                grid_template_rows="auto auto",
                grid_gap="10px",
                align_items="center",
                margin="0px 0px 5px 0px",
            ),
        )

        # Apply consistent width to output container and reduce its top margin
        self.output.layout.width = controls_width
        self.output.layout.margin = "0px"

        # Initialize widget visibility
        self._update_task_widgets()

        # Create a compact main container with minimal spacing
        return widgets.VBox(
            [controls, self.output],
            layout=widgets.Layout(
                width=controls_width,
                spacing="0px",
            ),
        )

    def display(self):
        """Display the widgets and initialize the plot."""
        # Create the layout
        layout = self.create_layout()

        # Display the layout
        display(layout)

        # Initialize the plot
        self._update_plot()
