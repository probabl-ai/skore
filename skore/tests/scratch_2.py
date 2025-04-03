# %% [markdown]
#
# Generate a data for a dataframe that could be the available metadata for a given
# experiment.

# %%
from typing import Literal, Union

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import clear_output, display
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import scikit-learn models
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from skore import ComparisonReport, EstimatorReport
from plotly.graph_objects import FigureWidget

rng = np.random.default_rng(42)

size = 100
index_reg_vs_clf = rng.choice([True, False], size=size)
ml_task = ["classification" if idx else "regression" for idx in index_reg_vs_clf]

rmse = rng.uniform(0, 100, size=size)
rmse[index_reg_vs_clf] = np.nan
r2_score = rng.uniform(0, 1, size=size)
r2_score[index_reg_vs_clf] = np.nan

log_loss = rng.uniform(0, 100, size=size)
accuracy_score = rng.uniform(0, 1, size=size)
precision_score = rng.uniform(0, 1, size=size)
recall_score = rng.uniform(0, 1, size=size)

log_loss[~index_reg_vs_clf] = np.nan
accuracy_score[~index_reg_vs_clf] = np.nan
precision_score[~index_reg_vs_clf] = np.nan
recall_score[~index_reg_vs_clf] = np.nan

regressor = rng.choice(["Ridge", "RandomForestRegressor"], size=size)
classifier = rng.choice(["LogisticRegression", "RandomForestClassifier"], size=size)
learner = np.where(index_reg_vs_clf, classifier, regressor)

# Create scalers with constraint: RandomForest always has None as scaler
possible_scalers = ["StandardScaler", "MinMaxScaler", "None"]
scaler = np.array([rng.choice(possible_scalers) for _ in range(size)])

# For RandomForest models, set scaler to None
is_random_forest = np.array(
    [(l == "RandomForestRegressor" or l == "RandomForestClassifier") for l in learner]
)
scaler[is_random_forest] = "None"

# Create encoders randomly
possible_encoders = ["OneHotEncoder", "None", "OrdinalEncoder"]
encoder = np.array([rng.choice(possible_encoders) for _ in range(size)])

# Generate dataset hash-like identifiers (2 for regression, 2 for classification)
reg_datasets = ["reg_dataset_" + hex(rng.integers(10000, 99999))[2:] for _ in range(2)]
clf_datasets = ["clf_dataset_" + hex(rng.integers(10000, 99999))[2:] for _ in range(2)]

# Assign datasets based on task type
dataset = []
for is_clf in index_reg_vs_clf:
    if is_clf:
        dataset.append(rng.choice(clf_datasets))
    else:
        dataset.append(rng.choice(reg_datasets))

# Generate random fit and predict times
fit_time = rng.uniform(0.01, 5.0, size=size)  # between 10ms and 5s
predict_time = rng.uniform(0.001, 0.5, size=size)  # between 1ms and 500ms

data = {
    "dataset": dataset,
    "ml_task": ml_task,
    "learner": learner,
    "scaler": scaler,
    "encoder": encoder,
    "r2_score": r2_score,
    "rmse": rmse,
    "accuracy_score": accuracy_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "log_loss": log_loss,
    "fit_time": fit_time,
    "predict_time": predict_time,
}

# %%
class MetaDataFrame(pd.DataFrame):
    """
    Enhanced DataFrame for interactive visualization of ML experiment metadata.

    This class extends pandas DataFrame with functionality for creating interactive
    parallel coordinates plots, enabling visual filtering and selection of
    machine learning experiments based on various metrics.
    """

    _metadata = [
        "_dimension_to_column",
        "_column_to_dimension",
        "_current_fig",
        "_current_dimensions",
        "_current_selection",
        "_task_dropdown",
        "_dataset_dropdown",
        "_metric_checkboxes",
        "_color_metric_dropdown",
        "_output",
        "_invert_colormap",
        "_clf_datasets",
        "_reg_datasets",
    ]

    def __init__(self, *args, **kwargs):
        """
        Initialize a new MetaDataFrame instance.
        """
        super().__init__(*args, **kwargs)

        # Define mapping between display names and column names
        self._dimension_to_column = {
            "Dataset": "dataset",
            "Learner": "learner",
            "Scaler": "scaler",
            "Encoder": "encoder",
            "ML Task": "ml_task",
            "R2 Score": "r2_score",
            "RMSE": "rmse",
            "Accuracy Score": "accuracy_score",
            "Precision Score": "precision_score",
            "Recall Score": "recall_score",
            "Log Loss": "log_loss",
            "Fit Time": "fit_time",
            "Predict Time": "predict_time",
        }
        self._column_to_dimension = {v: k for k, v in self._dimension_to_column.items()}

        # Define metrics where lower values are better (for inverted colormap)
        self._invert_colormap = ["RMSE", "Log Loss"]

        # Store the current figure, dimension filters, and selections
        self._current_fig = None
        self._current_dimensions = None
        self._current_selection = {}

        # Extract dataset information from input data instead of filtering self
        # This prevents recursion issues
        if args and isinstance(args[0], dict):
            input_data = args[0]
            ml_task_array = np.array(input_data.get('ml_task', []))
            dataset_array = np.array(input_data.get('dataset', []))

            # Get unique datasets for classification and regression tasks
            self._clf_datasets = np.unique(dataset_array[ml_task_array == 'classification']).tolist()
            self._reg_datasets = np.unique(dataset_array[ml_task_array == 'regression']).tolist()
        else:
            # Fallback to empty lists if we can't determine from input
            self._clf_datasets = []
            self._reg_datasets = []

        # Define metric sets with fit_time and predict_time
        classification_metrics = ["Accuracy Score", "Precision Score", "Recall Score", "Log Loss", "Fit Time", "Predict Time"]
        regression_metrics = ["R2 Score", "RMSE", "Fit Time", "Predict Time"]

        # Create task dropdown
        self._task_dropdown = widgets.Dropdown(
            options=[("Classification", "classification"), ("Regression", "regression")],
            value="classification",
            description="Task Type:",
            disabled=False,
            layout=widgets.Layout(width='200px')
        )

        # Create dataset dropdown that will update based on task
        self._dataset_dropdown = widgets.Dropdown(
            options=self._clf_datasets,  # Default to classification datasets
            description="Dataset:",
            disabled=False,
            layout=widgets.Layout(width='250px')
        )

        # Initialize metric checkboxes as dictionaries
        self._metric_checkboxes = {
            "classification": {},
            "regression": {}
        }

        # Create classification metric checkboxes
        for metric in classification_metrics:
            self._metric_checkboxes["classification"][metric] = widgets.Checkbox(
                value=True,
                description=metric,
                disabled=False,
                layout=widgets.Layout(width='auto')
            )

        # Create regression metric checkboxes
        for metric in regression_metrics:
            self._metric_checkboxes["regression"][metric] = widgets.Checkbox(
                value=True,
                description=metric,
                disabled=False,
                layout=widgets.Layout(width='auto')
            )

        # Create color metric dropdowns
        self._color_metric_dropdown = {
            "classification": widgets.Dropdown(
                options=classification_metrics,
                value="Log Loss",  # Default for classification
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(width='200px')
            ),
            "regression": widgets.Dropdown(
                options=regression_metrics,
                value="RMSE",  # Default for regression
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(width='200px')
            )
        }

        # Create containers for metric checkboxes
        self._classification_metrics_box = widgets.HBox(
            [widgets.Label("Metrics:")] +
            [self._metric_checkboxes["classification"][m] for m in classification_metrics]
        )

        self._regression_metrics_box = widgets.HBox(
            [widgets.Label("Metrics:")] +
            [self._metric_checkboxes["regression"][m] for m in regression_metrics]
        )

        # Set up callbacks
        self._task_dropdown.observe(self._on_task_change, names="value")
        self._dataset_dropdown.observe(self._update_plot, names="value")

        # Set up metric checkbox callbacks
        for task in ["classification", "regression"]:
            for metric in self._metric_checkboxes[task]:
                self._metric_checkboxes[task][metric].observe(self._update_plot, names="value")
            self._color_metric_dropdown[task].observe(self._update_plot, names="value")

        # Output area for the plot
        self._output = widgets.Output()

    def _on_task_change(self, change):
        """Handle task dropdown change event."""
        task = change["new"]

        # Update dataset dropdown options based on task
        if task == "classification":
            self._dataset_dropdown.options = self._clf_datasets
            if self._clf_datasets:
                self._dataset_dropdown.value = self._clf_datasets[0]
        else:
            self._dataset_dropdown.options = self._reg_datasets
            if self._reg_datasets:
                self._dataset_dropdown.value = self._reg_datasets[0]

        # Update UI visibility
        self._update_task_widgets()

        # Update the plot
        self._update_plot()

    def _update_task_widgets(self):
        """Update widget visibility based on selected task."""
        task = self._task_dropdown.value

        if task == "classification":
            self._classification_metrics_box.layout.display = ''
            self._color_metric_dropdown["classification"].layout.display = ''

            self._regression_metrics_box.layout.display = 'none'
            self._color_metric_dropdown["regression"].layout.display = 'none'
        else:
            self._classification_metrics_box.layout.display = 'none'
            self._color_metric_dropdown["classification"].layout.display = 'none'

            self._regression_metrics_box.layout.display = ''
            self._color_metric_dropdown["regression"].layout.display = ''

    @property
    def _constructor(self):
        """Return the constructor for this class."""
        return MetaDataFrame

    def _add_jitter(self, values, amount=0.05):
        """Add jitter to categorical values to improve visualization."""
        categories = pd.Categorical(values).codes
        return categories + rng.uniform(-amount, amount, size=len(values))

    def _update_plot(self, change=None):
        """Update the parallel coordinates plot based on the selected options."""
        with self._output:
            clear_output(wait=True)

            task = self._task_dropdown.value

            if not hasattr(self._dataset_dropdown, 'value') or not self._dataset_dropdown.value:
                display(widgets.HTML("No dataset available for selected task."))
                return

            dataset_name = self._dataset_dropdown.value

            # Filter data for the selected dataset
            filtered_df = self[self["dataset"] == dataset_name].copy()

            if filtered_df.empty:
                display(widgets.HTML(f"No data available for dataset: {dataset_name}"))
                return

            # Get selected metrics and color metric based on task
            if task == "classification":
                available_metrics = ["Accuracy Score", "Precision Score", "Recall Score", "Log Loss", "Fit Time", "Predict Time"]
                selected_metrics = [m for m in available_metrics if self._metric_checkboxes[task][m].value]
                color_metric = self._color_metric_dropdown[task].value
            else:
                available_metrics = ["R2 Score", "RMSE", "Fit Time", "Predict Time"]
                selected_metrics = [m for m in available_metrics if self._metric_checkboxes[task][m].value]
                color_metric = self._color_metric_dropdown[task].value

            # Debug selected metrics
            print(f"Selected metrics: {selected_metrics}")

            # Convert display names to column names
            selected_columns = [self._dimension_to_column[m] for m in selected_metrics]
            color_column = self._dimension_to_column[color_metric]

            # Create dimensions list in the required order:
            # 1. Learner
            # 2. Fit/Predict time metrics
            # 3. Statistical metrics
            dimensions = []

            # 1. Add learner with jitter
            filtered_df["learner_jittered"] = self._add_jitter(filtered_df["learner"])
            dimensions.append(
                dict(
                    label="Learner",
                    values=filtered_df["learner_jittered"],
                    ticktext=filtered_df["learner"].unique().tolist(),
                    tickvals=self._add_jitter(filtered_df["learner"].unique(), amount=0)
                )
            )

            # Categorize metrics
            time_metrics = ["fit_time", "predict_time"]
            statistical_metrics = [col for col in selected_columns if col not in time_metrics]

            # 2. Add time metrics
            for col in time_metrics:
                if col in selected_columns and not pd.isna(filtered_df[col]).all():
                    dimensions.append(
                        dict(
                            label=self._column_to_dimension[col],
                            values=filtered_df[col].fillna(0).tolist()
                        )
                    )

            # 3. Add statistical metrics
            for col in statistical_metrics:
                if not pd.isna(filtered_df[col]).all():  # Only add if not all NaN
                    dimensions.append(
                        dict(
                            label=self._column_to_dimension[col],
                            values=filtered_df[col].fillna(0).tolist()
                        )
                    )

            # Create colorscale (invert for metrics where lower is better)
            colorscale = "Viridis_r" if color_metric in self._invert_colormap else "Viridis"

            # Create the figure
            fig = go.Figure(
                data=go.Parcoords(
                    line=dict(
                        color=filtered_df[color_column].fillna(0).tolist(),
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(title=color_metric)
                    ),
                    dimensions=dimensions
                )
            )

            fig.update_layout(
                title=f"Parallel Coordinates Plot - {dataset_name} ({task})",
                title_y=0.97,  # Move title higher
                height=600,
                margin=dict(l=150, r=150, t=100, b=30)  # Increased margins
            )

            # Store current figure and dimensions
            self._current_fig = fig
            self._current_dimensions = dimensions

            display(fig)

    def _repr_html_(self):
        """Display the interactive plot and controls."""
        # Create controls layout
        controls_row1 = widgets.HBox([
            self._task_dropdown,
            self._dataset_dropdown,
            self._color_metric_dropdown["classification"],
            self._color_metric_dropdown["regression"]
        ])

        # Organize metric checkboxes differently to make all visible
        # Improve layout with a VBox for classification metrics
        clf_metrics_layout = widgets.VBox([
            widgets.HTML("<b>Classification Metrics:</b>"),
            widgets.HBox([
                self._metric_checkboxes["classification"]["Accuracy Score"],
                self._metric_checkboxes["classification"]["Precision Score"],
                self._metric_checkboxes["classification"]["Recall Score"],
            ]),
            widgets.HBox([
                self._metric_checkboxes["classification"]["Log Loss"],
                self._metric_checkboxes["classification"]["Fit Time"],
                self._metric_checkboxes["classification"]["Predict Time"],
            ])
        ])

        # Improve layout with a VBox for regression metrics
        reg_metrics_layout = widgets.VBox([
            widgets.HTML("<b>Regression Metrics:</b>"),
            widgets.HBox([
                self._metric_checkboxes["regression"]["R2 Score"],
                self._metric_checkboxes["regression"]["RMSE"],
                self._metric_checkboxes["regression"]["Fit Time"],
                self._metric_checkboxes["regression"]["Predict Time"],
            ])
        ])

        # Replace old metrics boxes with new layout
        self._classification_metrics_box = clf_metrics_layout
        self._regression_metrics_box = reg_metrics_layout

        controls_row2 = widgets.HBox([
            self._classification_metrics_box,
            self._regression_metrics_box
        ])

        controls = widgets.VBox([controls_row1, controls_row2])

        # Initialize widget visibility
        self._update_task_widgets()

        # Display controls and plot
        display(controls)
        display(self._output)

        # Initialize the plot
        self._update_plot()

        return ""

# %%
df = MetaDataFrame(data)
df

# %%
