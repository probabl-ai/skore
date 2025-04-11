# %% [markdown]
#
# Generate a data for a dataframe that could be the available metadata for a given
# experiment.

# %%

import numpy as np
import pandas as pd

# Import scikit-learn models
from skore.project._metadata import ModelExplorerWidget


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
        "_invert_colormap",
        "_plot_widget",
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
            "RMSE": "rmse",
            "median Absolute Error": "median_absolute_error",
            "mean Average Precision": "mean_average_precision",
            "macro ROC AUC": "macro_roc_auc",
            "Log Loss": "log_loss",
            "Fit Time": "fit_time",
            "Predict Time": "predict_time",
        }
        self._column_to_dimension = {v: k for k, v in self._dimension_to_column.items()}

        # Define metrics where lower values are better (for inverted colormap)
        self._invert_colormap = [
            "RMSE",
            "Log Loss",
            "Fit Time",
            "Predict Time",
            "median Absolute Error",
        ]

    @property
    def _constructor(self):
        """Return the constructor for this class."""
        return MetaDataFrame

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
            dimension_to_column=self._dimension_to_column,
            column_to_dimension=self._column_to_dimension,
            invert_colormap=self._invert_colormap,
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
            elif dim_name in self._column_to_dimension.values():
                # Find the column name that corresponds to this dimension label
                col_name = None
                for col, label in self._column_to_dimension.items():
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

    def to_frame(self, filter=True):
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

        if not hasattr(self, "_plot_widget"):
            return df

        if filter:
            query_string = self.query_string_selection()
            if query_string:
                return df.query(query_string)

            # If no query string but we have a selected dataset and task, filter based
            # on those
            task = self._plot_widget.task_dropdown.value
            if (
                hasattr(self._plot_widget.dataset_dropdown, "value")
                and self._plot_widget.dataset_dropdown.value
            ):
                dataset_name = self._plot_widget.dataset_dropdown.value
                return df[(df["ml_task"] == task) & (df["dataset"] == dataset_name)]

            # Otherwise, just filter based on task
            return df[df["ml_task"] == task]

        return df


rng = np.random.default_rng(42)

size = 100
index_reg_vs_clf = rng.choice([True, False], size=size)
ml_task = ["classification" if idx else "regression" for idx in index_reg_vs_clf]

median_absolute_error = rng.uniform(0, 100, size=size)
median_absolute_error[index_reg_vs_clf] = np.nan
rmse = rng.uniform(0, 100, size=size)
rmse[index_reg_vs_clf] = np.nan

log_loss = rng.uniform(0, 100, size=size)
mean_average_precision = rng.uniform(0, 1, size=size)
macro_roc_auc = rng.uniform(0, 1, size=size)

log_loss[~index_reg_vs_clf] = np.nan
mean_average_precision[~index_reg_vs_clf] = np.nan
macro_roc_auc[~index_reg_vs_clf] = np.nan

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
    "median_absolute_error": median_absolute_error,
    "rmse": rmse,
    "mean_average_precision": mean_average_precision,
    "macro_roc_auc": macro_roc_auc,
    "log_loss": log_loss,
    "fit_time": fit_time,
    "predict_time": predict_time,
}

# %%
df = MetaDataFrame(data)

# %%
df

# %%
df.to_frame()

# %%
df.query_string_selection()

# %%
df

# %%
df.iloc[:3].to_frame(filter=False)

# %%
