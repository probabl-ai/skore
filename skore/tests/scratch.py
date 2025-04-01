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
}

# %%
from plotly.graph_objects import FigureWidget


class MetaDataFrame(pd.DataFrame):
    """
    Enhanced DataFrame for interactive visualization of ML experiment metadata.

    This class extends pandas DataFrame with functionality for creating interactive
    parallel coordinates plots, enabling visual filtering and selection of
    machine learning experiments based on various metrics.

    Parameters
    ----------
    *args : tuple
        Arguments to pass to pandas.DataFrame constructor.
    **kwargs : dict
        Keyword arguments to pass to pandas.DataFrame constructor.

    Attributes
    ----------
    _dimension_to_column : dict
        Mapping from display names to actual column names.
    _column_to_dimension : dict
        Mapping from column names to display names.
    _current_fig : plotly.graph_objects.FigureWidget or None
        The current figure being displayed.
    _current_dimensions : list or None
        List of dimension configurations for the parallel coordinates plot.
    _current_selection : dict
        Dictionary storing the current selection ranges for each dimension.
    _task_dropdown : ipywidgets.Dropdown
        Dropdown for selecting ML task (regression or classification).
    _metric_dropdown : ipywidgets.Dropdown
        Dropdown for selecting relevant performance metrics based on the task.
    _output : ipywidgets.Output
        Output area for the plot.
    _invert_colormap : list
        List of metrics that should have inverted colormaps (lower is better).
    _show_dataset_toggle : ipywidgets.Checkbox
        Checkbox for toggling dataset dimension visibility.

    Methods
    -------
    update_selection
        Explicitly updates the selection based on the current plot state.
    get_selection_query
        Generates a pandas query string based on the visual selection.
    to_frame
        Converts the MetaDataFrame to a standard pandas DataFrame.
    update_plot
        Updates the parallel coordinates plot based on selected task and metrics.

    Examples
    --------
    >>> df = MetaDataFrame(data)
    >>> df  # Displays interactive parallel coordinates plot with dropdowns
    >>> query = df.get_selection_query()
    >>> filtered_df = df.query(query)
    """

    _metadata = [
        "_dimension_to_column",
        "_column_to_dimension",
        "_current_fig",
        "_current_dimensions",
        "_current_selection",
        "_task_dropdown",
        "_metric_dropdown",
        "_output",
        "_invert_colormap",
        "_show_dataset_toggle",
    ]

    def __init__(self, *args, **kwargs):
        """
        Initialize a new MetaDataFrame instance.

        Parameters
        ----------
        *args : tuple
            Arguments to pass to pandas.DataFrame constructor.
        **kwargs : dict
            Keyword arguments to pass to pandas.DataFrame constructor.
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
        }
        self._column_to_dimension = {v: k for k, v in self._dimension_to_column.items()}

        # Define metrics where lower values are better (for inverted colormap)
        self._invert_colormap = ["RMSE", "Log Loss"]

        # Store the current figure, dimension filters, and selections
        self._current_fig = None
        self._current_dimensions = None
        self._current_selection = {}

        # Create dropdowns for task and metric selection
        self._task_dropdown = widgets.Dropdown(
            options=["Regression", "Classification"],
            value="Classification",
            description="ML Task:",
            disabled=False,
        )

        # Initialize metric dropdown with classification metrics - Log Loss as default
        self._metric_dropdown = widgets.Dropdown(
            options=["Log Loss", "Accuracy Score", "Precision Score", "Recall Score"],
            value="Log Loss",
            description="Metrics:",
            disabled=False,
        )

        # Create toggle for dataset dimension
        self._show_dataset_toggle = widgets.Checkbox(
            value=True, description="Show Dataset", disabled=False
        )

        # Set up callbacks
        self._task_dropdown.observe(self._on_task_change, names="value")
        self._metric_dropdown.observe(self._on_metric_change, names="value")
        self._show_dataset_toggle.observe(self._on_dataset_toggle_change, names="value")

        # Output area for the plot
        self._output = widgets.Output()

    def _on_task_change(self, change):
        """Handle task dropdown change event."""
        task = change["new"]

        # Update metric dropdown options based on selected task
        if task == "Regression":
            self._metric_dropdown.options = ["RMSE", "R2 Score"]
            self._metric_dropdown.value = "RMSE"
        elif task == "Classification":
            self._metric_dropdown.options = [
                "Log Loss",
                "Accuracy Score",
                "Precision Score",
                "Recall Score",
            ]
            self._metric_dropdown.value = "Log Loss"

        # Update the plot
        self.update_plot()

    def _on_metric_change(self, change):
        """Handle metric dropdown change event."""
        # Update the plot when metric selection changes
        self.update_plot()

    def _on_dataset_toggle_change(self, change):
        """Handle dataset toggle change event."""
        # Update the plot when dataset visibility changes
        self.update_plot()

    @property
    def _constructor(self):
        """
        Return the constructor for this class.

        This is required for pandas DataFrame subclasses to ensure proper
        inheritance during operations that return a new DataFrame.

        Returns
        -------
        type
            The MetaDataFrame class constructor.
        """
        return MetaDataFrame

    def update_plot(self):
        """
        Update the parallel coordinates plot based on the selected task and metrics.
        """
        with self._output:
            clear_output(wait=True)

            # Filter data based on task selection
            filtered_df = self

            if self._task_dropdown.value == "Regression":
                filtered_df = self[self["ml_task"] == "regression"]
            elif self._task_dropdown.value == "Classification":
                filtered_df = self[self["ml_task"] == "classification"]

            # Prepare columns for the plot based on metric selection
            columns_to_show = []

            # Include dataset as the leftmost dimension if toggle is on
            if self._show_dataset_toggle.value and "dataset" in self.columns:
                columns_to_show.append("dataset")

            # Include categorical dimensions in the desired order: scaler, encoder, learner
            if "scaler" in self.columns:
                columns_to_show.append("scaler")
            if "encoder" in self.columns:
                columns_to_show.append("encoder")
            if "learner" in self.columns:
                columns_to_show.append("learner")

            # Add metrics based on selection
            task = self._task_dropdown.value
            metric = self._metric_dropdown.value

            # Add the selected metric
            col_name = self._dimension_to_column[metric]
            columns_to_show.append(col_name)

            # Create dimensions list for parallel coordinates plot
            dimensions = []

            # Add each dimension with appropriate configuration
            for col in columns_to_show:
                if col in ["learner", "scaler", "encoder", "dataset"]:
                    # Get unique categorical values
                    unique_values = filtered_df[col].unique().tolist()
                    # Map each value to a normalized numerical index (0 to 1 range)
                    n_values = len(unique_values)
                    if n_values > 1:
                        # Map to evenly spaced points between 0.1 and 0.9
                        spacing = 0.8 / (n_values - 1)
                        value_to_index = {
                            value: 0.1 + i * spacing
                            for i, value in enumerate(sorted(unique_values))
                        }
                    else:
                        # If only one value, place it at 0.5
                        value_to_index = {unique_values[0]: 0.5}

                    # Add controlled jitter that keeps values within proper bounds
                    base_values = np.array(
                        [value_to_index[value] for value in filtered_df[col]]
                    )

                    # Apply asymmetric jitter that avoids the edges
                    jitter = np.zeros_like(base_values)

                    # Process each unique value to apply appropriate jitter
                    for value in unique_values:
                        mask = filtered_df[col] == value
                        val_position = value_to_index[value]

                        # For edge categories, use asymmetric jitter to avoid boundaries
                        if val_position <= 0.1:  # Lowest category
                            # Only apply positive jitter
                            jitter[mask] = rng.uniform(0.01, 0.03, size=mask.sum())
                        elif val_position >= 0.9:  # Highest category
                            # Only apply negative jitter
                            jitter[mask] = rng.uniform(-0.03, -0.01, size=mask.sum())
                        else:  # Middle categories
                            # Apply balanced jitter
                            jitter_range = min(0.03, spacing / 3)
                            jitter[mask] = rng.uniform(
                                -jitter_range, jitter_range, size=mask.sum()
                            )

                    values = base_values + jitter

                    # Additional safety: ensure values stay well within [0.02, 0.98] range for selection
                    values = np.clip(values, 0.02, 0.98).tolist()

                    # Create the dimension with standardized range
                    dimensions.append(
                        dict(
                            range=[0, 1],  # Standardized range for all dimensions
                            label=self._column_to_dimension.get(
                                col, col.replace("_", " ").title()
                            ),
                            values=values,
                            tickvals=[
                                value_to_index[val] for val in sorted(unique_values)
                            ],
                            ticktext=sorted(unique_values),
                        )
                    )
                elif col in ["rmse", "log_loss"]:
                    # For RMSE and Log Loss, use actual values (not normalized)
                    values = filtered_df[col].fillna(0).tolist()

                    # Get min and max for the range with some padding
                    if filtered_df[col].dropna().size > 0:
                        min_val = 0  # Always start at 0 for these metrics
                        max_val = (
                            filtered_df[col].dropna().max() * 1.1
                        )  # Add 10% margin
                    else:
                        min_val, max_val = 0, 1

                    dimensions.append(
                        dict(
                            range=[min_val, max_val],
                            label=self._column_to_dimension.get(
                                col, col.replace("_", " ").title()
                            ),
                            values=values,
                        )
                    )
                else:
                    # For other numerical columns (already in 0-1 range)
                    values = filtered_df[col].fillna(0).tolist()
                    dimensions.append(
                        dict(
                            range=[0, 1],  # All metrics are between 0 and 1
                            label=self._column_to_dimension.get(
                                col, col.replace("_", " ").title()
                            ),
                            values=values,
                        )
                    )

            # Get the selected metric column for coloring
            color_column = self._dimension_to_column[metric]
            color_values = filtered_df[color_column].fillna(0).tolist()

            # Use viridis colormap for all tasks, inverted for RMSE and Log Loss
            colorscale = "viridis"
            if metric in self._invert_colormap:
                colorscale = "viridis_r"  # Inverted viridis

            colorbar = dict(
                title=metric,
                thickness=15,
                x=1.05,  # Move colorbar further to the right
                xpad=10,  # Add padding between plot and colorbar
            )

            # Create the figure
            regular_fig = go.Figure(
                data=go.Parcoords(
                    line=dict(
                        color=color_values,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=colorbar,
                    ),
                    dimensions=dimensions,
                )
            )

            # Convert to FigureWidget for interactivity
            fig = FigureWidget(regular_fig)

            fig.update_layout(
                title=(
                    f"Parallel Coordinates Plot for {self._task_dropdown.value} Task "
                    f"- {self._metric_dropdown.value}"
                ),
                title_x=0.5,  # Center the title
                title_y=0.97,  # Position the title higher
                font=dict(size=14),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=600,
                width=900,
                margin=dict(
                    l=150,
                    r=180,
                    t=120,
                    b=80,  # Increased top margin
                ),  # Significantly increased right margin for colorbar
            )

            # Store the current figure and dimensions for later use
            self._current_fig = fig
            self._current_dimensions = dimensions

            display(fig)

    def _repr_html_(self):
        """
        Create and display an interactive parallel coordinates plot with dropdown controls.

        This method is automatically called when the DataFrame is displayed
        in a Jupyter notebook. It creates dropdown menus for task and metric selection,
        along with a parallel coordinates plot.

        Returns
        -------
        str
            An empty string, as the display is handled by IPython.display.
        """
        # Create a container for the controls and plot
        controls = widgets.HBox(
            [self._task_dropdown, self._metric_dropdown, self._show_dataset_toggle]
        )
        display(widgets.VBox([controls, self._output]))

        # Initialize the plot
        self.update_plot()

        return ""

    def update_selection(self):
        """
        Update the selection based on the current state of the plot.

        This method explicitly captures the current selection state from
        the parallel coordinates plot. It should be called when you want
        to capture the current visual selection before querying.

        Returns
        -------
        MetaDataFrame
            Self reference for method chaining.
        """
        if not self._current_fig or not hasattr(
            self._current_fig.data[0], "dimensions"
        ):
            return self

        # Extract the constraint ranges from the plot data
        selection_data = {}
        for i, dim in enumerate(self._current_fig.data[0].dimensions):
            if hasattr(dim, "constraintrange") and dim.constraintrange:
                dim_name = self._current_dimensions[i]["label"]
                selection_data[dim_name] = dim.constraintrange

        print(f"Current selection: {selection_data}")
        self._current_selection = selection_data
        return self

    def get_selection_query(self):
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
        >>> query_string = df.get_selection_query()
        >>> filtered_df = df.query(query_string)
        """
        # First update the selection to ensure we have the latest state
        self.update_selection()

        # Build the query string based on current selections and task filter
        if not self._current_selection:
            # Always apply the task filter
            task_value = self._task_dropdown.value.lower()
            return f"ml_task == '{task_value}'"

        query_parts = []

        # Always add task filter
        task_value = self._task_dropdown.value.lower()
        query_parts.append(f"ml_task == '{task_value}'")

        for dim_name, range_values in self._current_selection.items():
            # Handle categorical dimensions
            if dim_name in ["Learner", "Scaler", "Encoder", "Dataset"]:
                # Get the tick values and text for the dimension
                categorical_dim = None
                for dim in self._current_dimensions:
                    if dim["label"] == dim_name:
                        categorical_dim = dim
                        break

                if (
                    categorical_dim
                    and "ticktext" in categorical_dim
                    and "tickvals" in categorical_dim
                ):
                    # Find which values fall within the range
                    selected_values = []
                    min_val, max_val = range_values

                    for i, val in enumerate(categorical_dim["tickvals"]):
                        if min_val <= val <= max_val:
                            selected_values.append(categorical_dim["ticktext"][i])

                    if selected_values:
                        col_name = self._dimension_to_column[dim_name]
                        values_str = ", ".join(
                            [f"'{value}'" for value in selected_values]
                        )
                        query_parts.append(f"{col_name}.isin([{values_str}])")
            # Handle ML Task
            elif dim_name == "ML Task":
                if range_values[0] >= 0 and range_values[1] <= 0.5:
                    query_parts.append("ml_task == 'regression'")
                elif range_values[0] > 0.5 and range_values[1] <= 1:
                    query_parts.append("ml_task == 'classification'")
                # If selection spans both categories
                elif range_values[0] <= 0.5 and range_values[1] >= 0.5:
                    # No constraint needed as it includes both categories
                    pass
            else:
                # Handle numerical dimensions - use the actual column values for range
                # queries
                col_name = self._dimension_to_column.get(dim_name)
                if col_name:
                    # Check if we need to apply normalization for the query
                    min_val, max_val = range_values

                    # Format with appropriate precision
                    query_parts.append(
                        f"({col_name} >= {min_val:.6f} and {col_name} <= {max_val:.6f})"
                    )

        # Join all query parts with logical AND
        if query_parts:
            return " and ".join(query_parts)
        else:
            return ""

    def filter(self) -> "MetaDataFrame":
        """
        Filter the dataframe based on the current selection.

        Returns
        -------
        MetaDataFrame
            A filtered MetaDataFrame containing the same data.
        """
        query_string = self.get_selection_query()
        if query_string:
            return self.query(query_string)
        return self

    def to_frame(self, filter: bool = True) -> pd.DataFrame:
        """
        Convert this MetaDataFrame to a regular pandas DataFrame.

        This method creates a standard pandas DataFrame containing the same
        data as this MetaDataFrame, without the enhanced visualization
        capabilities.

        Parameters
        ----------
        filter : bool, default=True
            If True, the DataFrame will be filtered according to the current
            selection in the parallel coordinates plot.

        Returns
        -------
        pandas.DataFrame
            A standard pandas DataFrame containing the same data.
        """
        df = pd.DataFrame(self)
        if filter is not None and (query_string := self.get_selection_query()):
            return df.query(query_string)
        return df

    def reports(
        self,
        filter: bool = True,
        return_as: Literal["list", "dict", "ComparisonReport"] = "ComparisonReport",
    ) -> Union[list, dict[str, EstimatorReport], ComparisonReport]:
        """
        Return reports related to the current selection. The reports can be as a
        ComparisonReport.

        Parameters
        ----------
        filter : bool, default=True
            If True, the DataFrame will be filtered according to the current
            selection in the parallel coordinates plot.

        return_as : {"list", "dict", "ComparisonReport"}, default="ComparisonReport"
            The type of object to return.
        """
        df = self if not filter else self.filter()
        # FIXME: this is a mock implementation
        # Create a dictionary to map model strings to their class constructors
        model_classes = {
            "Ridge": Ridge,
            "RandomForestRegressor": RandomForestRegressor,
            "LogisticRegression": LogisticRegression,
            "RandomForestClassifier": RandomForestClassifier,
        }

        # Create a dictionary to map scaler strings to their class constructors
        scaler_classes = {
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "None": None,
        }

        # Create a dictionary to map encoder strings to their class constructors
        encoder_classes = {
            "OneHotEncoder": OneHotEncoder,
            "OrdinalEncoder": OrdinalEncoder,
            "None": None,
        }

        # Initialize result containers
        reports_list = []
        reports_dict = {}

        # Get all unique datasets and tasks in the filtered dataframe
        unique_datasets = df["dataset"].unique()

        # Generate dataset cache: mapping from dataset name to (X, y) data
        dataset_cache = {}

        # Generate a dataset for each unique dataset name
        n_samples, n_features = 100, 5

        for dataset_name in unique_datasets:
            # Get the task for this dataset
            dataset_rows = df[df["dataset"] == dataset_name]
            # All rows with the same dataset should have the same task
            task = dataset_rows["ml_task"].iloc[0]

            # Set a seed based on the dataset name (for reproducibility)
            dataset_seed = sum(ord(c) for c in dataset_name) % 10000
            dataset_rng = np.random.RandomState(dataset_seed)

            # Generate X data
            X = dataset_rng.rand(n_samples, n_features)

            # Generate appropriate labels based on task type
            if task == "regression":
                y = dataset_rng.rand(n_samples) * 100  # Scale to make it more realistic
            else:  # classification
                y = dataset_rng.choice([0, 1], size=n_samples)

            # Cache the dataset
            dataset_cache[dataset_name] = {"X": X, "y": y, "task": task}

        # Process each unique combination in the filtered DataFrame
        for idx, row in df.iterrows():
            # Get model, scaler, and encoder information
            learner_name = row["learner"]
            scaler_name = row["scaler"]
            encoder_name = row["encoder"]
            task = row["ml_task"]
            dataset_name = row["dataset"]

            # Create a unique identifier for this configuration
            config_id = f"{dataset_name}_{learner_name}_{scaler_name}_{encoder_name}"

            # Get the dataset from the cache
            if dataset_name not in dataset_cache:
                print(f"Warning: Dataset {dataset_name} not found in cache.")
                continue

            dataset = dataset_cache[dataset_name]
            X, y = dataset["X"], dataset["y"]

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            try:
                # Instantiate the model based on the learner name
                model_class = model_classes[learner_name]

                # Create preprocessing steps based on scaler and encoder
                preprocessing_steps = []

                if scaler_name != "None" and scaler_classes[scaler_name] is not None:
                    scaler = scaler_classes[scaler_name]()
                    preprocessing_steps.append(("scaler", scaler))

                if encoder_name != "None" and encoder_classes[encoder_name] is not None:
                    # In a real scenario, you'd apply the encoder only to categorical
                    # columns Here we're just mocking it
                    encoder = encoder_classes[encoder_name]()
                    if encoder_name == "OneHotEncoder":
                        encoder.set_params(handle_unknown="ignore")
                    else:
                        encoder.set_params(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        )
                    preprocessing_steps.append(("encoder", encoder))

                # Create the final model (pipeline or just the model)
                if preprocessing_steps:
                    # Use a pipeline if preprocessing is needed
                    pipeline = Pipeline(
                        preprocessing_steps + [("model", model_class())]
                    )
                    estimator = pipeline
                else:
                    # Use the model directly if no preprocessing
                    estimator = model_class()

                # Fit the model
                estimator.fit(X_train, y_train)

                # Create an EstimatorReport
                report = EstimatorReport(
                    estimator=estimator,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                )

                # Add to result containers
                reports_list.append(report)
                reports_dict[config_id] = report

            except Exception as e:
                print(f"Error creating report for {config_id}: {e}")
                continue

        # Return the appropriate container based on user preference
        if return_as == "list":
            return reports_list
        elif return_as == "dict":
            return reports_dict
        else:  # ComparisonReport
            if reports_dict:
                return ComparisonReport(reports_dict)
            else:
                raise ValueError(
                    "No valid reports could be generated for the selection."
                )


# %%
df = MetaDataFrame(data)
df

# %%
comparison_report = df.reports(filter=True)
comparison_report.metrics.report_metrics()

# %%
roc_display = comparison_report.metrics.roc()
roc_display.plot()
_ = roc_display.ax_.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# %%
