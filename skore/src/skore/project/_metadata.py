import numpy as np
import plotly.graph_objects as go
from IPython.display import clear_output, display
from ipywidgets import widgets


class ModelExplorerWidget:
    """
    Widget for interactive parallel coordinate plots of ML experiment metadata.

    This class handles the creation and management of interactive widgets
    and the parallel coordinate plot for exploring ML experiment data.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe containing the experiment metadata.
    seed : int, default=0
        Seed for the jittering categorical columns.
    """

    _plot_width = 800
    _dimension_to_column = {
        "Learner": "learner",
        "RMSE": "rmse",
        "median Absolute Error": "median_absolute_error",
        "mean Average Precision": "mean_average_precision",
        "macro ROC AUC": "macro_roc_auc",
        "Log Loss": "log_loss",
        "Fit Time": "fit_time",
        "Predict Time": "predict_time",
    }
    _column_to_dimension = {v: k for k, v in _dimension_to_column.items()}
    _invert_colormap = [
        "RMSE",
        "Log Loss",
        "Fit Time",
        "Predict Time",
        "median Absolute Error",
    ]

    def __init__(self, dataframe, seed=0):
        self.df = dataframe
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

        self._task_dropdown = widgets.Dropdown(
            options=[
                ("Classification", "classification"),
                ("Regression", "regression"),
            ],
            value="classification",
            description="Task Type:",
            disabled=False,
            layout=widgets.Layout(width="200px"),
        )

        self._dataset_dropdown = widgets.Dropdown(
            options=self._clf_datasets,
            description="Dataset:",
            disabled=False,
            layout=widgets.Layout(width="250px"),
        )

        self._metric_checkboxes = {"classification": {}, "regression": {}}
        for metric in time_metrics + classification_metrics:
            default_value = metric not in time_metrics
            self._metric_checkboxes["classification"][metric] = widgets.Checkbox(
                indent=False,
                value=default_value,
                description=metric,
                disabled=False,
                layout=widgets.Layout(width="auto", margin="0px 10px 0px 0px"),
            )
        for metric in time_metrics + regression_metrics:
            default_value = metric not in time_metrics
            self._metric_checkboxes["regression"][metric] = widgets.Checkbox(
                indent=False,
                value=default_value,
                description=metric,
                disabled=False,
                layout=widgets.Layout(width="auto", margin="0px 10px 0px 0px"),
            )
        self._color_metric_dropdown = {
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
                self._metric_checkboxes["classification"][metric]
                for metric in time_metrics + classification_metrics
            ]
        )
        self.regression_metrics_box = widgets.HBox(
            [
                self._metric_checkboxes["regression"][metric]
                for metric in time_metrics + regression_metrics
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
                self._metric_checkboxes["classification"]["Fit Time"],
                self._metric_checkboxes["classification"]["Predict Time"],
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
                self._metric_checkboxes["classification"]["mean Average Precision"],
                self._metric_checkboxes["classification"]["macro ROC AUC"],
                self._metric_checkboxes["classification"]["Log Loss"],
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
                self._metric_checkboxes["regression"]["Fit Time"],
                self._metric_checkboxes["regression"]["Predict Time"],
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
                self._metric_checkboxes["regression"]["median Absolute Error"],
                self._metric_checkboxes["regression"]["RMSE"],
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
        self.layout = widgets.VBox(
            [controls, self.output],
            layout=widgets.Layout(width=f"{self._plot_width}px", spacing="0px"),
        )

    def _on_task_change(self, change):
        """Handle task dropdown change event."""
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

    def _update_task_widgets(self):
        """Update widget visibility based on selected task."""
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

    @staticmethod
    def _add_jitter_to_categorical(seed, categorical_series, amount=0.01):
        """Add jitter to categorical values to improve visualization."""
        rng = np.random.default_rng(seed)
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

            task = self._task_dropdown.value

            if (
                not hasattr(self._dataset_dropdown, "value")
                or not self._dataset_dropdown.value
            ):
                display(widgets.HTML("No dataset available for selected task."))
                return

            df_dataset = self.df.query(
                "dataset == @self._dataset_dropdown.value"
            ).copy()
            for col in df_dataset.select_dtypes(include=["category"]).columns:
                df_dataset[col] = df_dataset[col].cat.remove_unused_categories()

            selected_metrics = [
                metric
                for metric in self._metric_checkboxes[task]
                if self._metric_checkboxes[task][metric].value
            ]
            color_metric = self._color_metric_dropdown[task].value

            selected_columns = [
                self._dimension_to_column[metric] for metric in selected_metrics
            ]
            color_column = self._dimension_to_column[color_metric]

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

            for col in selected_columns:  # use the order defined in the constructor
                dimensions.append(
                    dict(
                        label=self._column_to_dimension[col],
                        values=df_dataset[col].fillna(0),
                    )
                )

            colorscale = (
                "Viridis_r" if color_metric in self._invert_colormap else "Viridis"
            )
            fig = go.FigureWidget(
                data=go.Parcoords(
                    line=dict(
                        color=df_dataset[color_column].fillna(0),
                        colorscale=colorscale,
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

            fig.data[0].on_selection(self.update_selection)

            self.current_fig = fig
            self.current_dimensions = dimensions

            display(fig)

    def update_selection(self, trace=None, points=None, selector=None):
        """Callback for when the selection on the parallel coordinates plot changes.

        Parameters
        ----------
        trace : go.parcoords.Trace, optional
            The trace that changed.
        points : list of int, optional
            The points that changed.
        selector : dict, optional
            The selector that changed.

        Returns
        -------
        self
        """

        selection_data = {
            "ml_task": self._task_dropdown.value,
            "dataset": self._dataset_dropdown.value,
        }
        selection_data.update(
            {
                self._dimension_to_column[dim.label]: dim.constraintrange
                for dim in self.current_fig.data[0].dimensions
                if hasattr(dim, "constraintrange") and dim.constraintrange
            }
        )
        self.current_selection = selection_data

        return self

    def display(self):
        """Display the widgets and initialize the plot."""
        display(self.layout)
        self._update_plot()
        self.update_selection()
