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


class ParallelCoordinatePlotWidget:
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
    """

    def __init__(
        self,
        dataframe,
        dimension_to_column,
        column_to_dimension,
        invert_colormap,
        clf_datasets,
        reg_datasets,
    ):
        self.df = dataframe
        self.dimension_to_column = dimension_to_column
        self.column_to_dimension = column_to_dimension
        self.invert_colormap = invert_colormap
        self.clf_datasets = clf_datasets
        self.reg_datasets = reg_datasets

        # Initialize attributes
        self.current_fig = None
        self.current_dimensions = None
        self.current_selection = {}

        # Define metric sets with fit_time and predict_time
        self.classification_metrics = [
            "mean Average Precision",
            "macro ROC AUC",
            "Log Loss",
            "Fit Time",
            "Predict Time",
        ]
        self.regression_metrics = ["MedAE", "RMSE", "Fit Time", "Predict Time"]

        # Initialize widgets
        self._create_widgets()

        # Set up callbacks
        self._setup_callbacks()

        # Output area for the plot
        self.output = widgets.Output()

    def _create_widgets(self):
        """Create all the necessary widgets."""
        # Create task dropdown
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

        # Create dataset dropdown that will update based on task
        self.dataset_dropdown = widgets.Dropdown(
            options=self.clf_datasets,  # Default to classification datasets
            description="Dataset:",
            disabled=False,
            layout=widgets.Layout(width="250px"),
        )

        # Initialize metric checkboxes as dictionaries
        self.metric_checkboxes = {"classification": {}, "regression": {}}

        # Create classification metric checkboxes
        for metric in self.classification_metrics:
            # Set time metrics to be unchecked by default
            default_value = metric not in ["Fit Time", "Predict Time"]
            self.metric_checkboxes["classification"][metric] = widgets.Checkbox(
                indent=False,
                value=default_value,
                description=metric,
                disabled=False,
                layout=widgets.Layout(width="auto", margin="0px 20px 0px 0px"),
            )

        # Create regression metric checkboxes
        for metric in self.regression_metrics:
            # Set time metrics to be unchecked by default
            default_value = metric not in ["Fit Time", "Predict Time"]
            self.metric_checkboxes["regression"][metric] = widgets.Checkbox(
                indent=False,
                value=default_value,
                description=metric,
                disabled=False,
                layout=widgets.Layout(width="auto", margin="0px 20px 0px 0px"),
            )

        # Create color metric dropdowns
        self.color_metric_dropdown = {
            "classification": widgets.Dropdown(
                options=self.classification_metrics,
                value="Log Loss",  # Default for classification
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(width="200px"),
            ),
            "regression": widgets.Dropdown(
                options=self.regression_metrics,
                value="RMSE",  # Default for regression
                description="Color by:",
                disabled=False,
                layout=widgets.Layout(width="200px"),
            ),
        }

        # Create containers for metric checkboxes
        self.classification_metrics_box = widgets.HBox(
            [
                self.metric_checkboxes["classification"][m]
                for m in self.classification_metrics
            ]
        )

        self.regression_metrics_box = widgets.HBox(
            [self.metric_checkboxes["regression"][m] for m in self.regression_metrics]
        )

    def _setup_callbacks(self):
        """Set up widget callbacks."""
        # Task dropdown callback
        self.task_dropdown.observe(self._on_task_change, names="value")

        # Dataset dropdown callback
        self.dataset_dropdown.observe(self._update_plot, names="value")

        # Metric checkbox callbacks
        for task in ["classification", "regression"]:
            for metric in self.metric_checkboxes[task]:
                self.metric_checkboxes[task][metric].observe(
                    self._update_plot, names="value"
                )
            # Color metric dropdown callback
            self.color_metric_dropdown[task].observe(self._update_plot, names="value")

    def _on_task_change(self, change):
        """Handle task dropdown change event."""
        task = change["new"]

        # Update dataset dropdown options based on task
        if task == "classification":
            self.dataset_dropdown.options = self.clf_datasets
            if self.clf_datasets:
                self.dataset_dropdown.value = self.clf_datasets[0]
        else:
            self.dataset_dropdown.options = self.reg_datasets
            if self.reg_datasets:
                self.dataset_dropdown.value = self.reg_datasets[0]

        # Update UI visibility
        self._update_task_widgets()

        # Update the plot
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

    def _add_jitter(self, values, amount=0.05):
        """
        Add jitter to categorical values to improve visualization.

        Applies asymmetric jitter for edge values to ensure selection works properly:
        - Lowest category gets only positive jitter
        - Highest category gets only negative jitter
        - Middle categories get balanced jitter
        """
        # Convert to categorical codes (0, 1, 2, etc.)
        categories = pd.Categorical(values).codes

        # Get unique category codes
        unique_codes = np.unique(categories)
        min_code = unique_codes.min()
        max_code = unique_codes.max()

        # Initialize jitter array
        jitter = np.zeros_like(categories, dtype=float)

        # Apply appropriate jitter based on position
        for code in unique_codes:
            mask = categories == code

            if code == min_code:  # Bottom category
                # Apply only positive jitter
                jitter[mask] = np.random.uniform(0, amount, size=mask.sum())
            elif code == max_code:  # Top category
                # Apply only negative jitter
                jitter[mask] = np.random.uniform(-amount, 0, size=mask.sum())
            else:  # Middle categories
                # Apply balanced jitter
                jitter[mask] = np.random.uniform(-amount, amount, size=mask.sum())

        return categories + jitter

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

            dataset_name = self.dataset_dropdown.value

            # Filter data for the selected dataset
            filtered_df = self.df[self.df["dataset"] == dataset_name].copy()

            if filtered_df.empty:
                display(widgets.HTML(f"No data available for dataset: {dataset_name}"))
                return

            # Get selected metrics and color metric based on task
            if task == "classification":
                available_metrics = self.classification_metrics
                selected_metrics = [
                    m
                    for m in available_metrics
                    if self.metric_checkboxes[task][m].value
                ]
                color_metric = self.color_metric_dropdown[task].value
            else:
                available_metrics = self.regression_metrics
                selected_metrics = [
                    m
                    for m in available_metrics
                    if self.metric_checkboxes[task][m].value
                ]
                color_metric = self.color_metric_dropdown[task].value

            # Convert display names to column names
            selected_columns = [self.dimension_to_column[m] for m in selected_metrics]
            color_column = self.dimension_to_column[color_metric]

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
                    tickvals=self._add_jitter(
                        filtered_df["learner"].unique(), amount=0
                    ),
                )
            )

            # Categorize metrics
            time_metrics = ["fit_time", "predict_time"]
            statistical_metrics = [
                col for col in selected_columns if col not in time_metrics
            ]

            # 2. Add time metrics
            for col in time_metrics:
                if col in selected_columns and not pd.isna(filtered_df[col]).all():
                    dimensions.append(
                        dict(
                            label=self.column_to_dimension[col],
                            values=filtered_df[col].fillna(0).tolist(),
                        )
                    )

            # 3. Add statistical metrics
            for col in statistical_metrics:
                if not pd.isna(filtered_df[col]).all():  # Only add if not all NaN
                    dimensions.append(
                        dict(
                            label=self.column_to_dimension[col],
                            values=filtered_df[col].fillna(0).tolist(),
                        )
                    )

            # Create colorscale (invert for metrics where lower is better)
            colorscale = (
                "Viridis_r" if color_metric in self.invert_colormap else "Viridis"
            )

            # Create the figure
            fig = go.Figure(
                data=go.Parcoords(
                    line=dict(
                        color=filtered_df[color_column].fillna(0).tolist(),
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(title=color_metric),
                    ),
                    dimensions=dimensions,
                )
            )

            # Set consistent width and layout
            plot_width = 800  # Width in pixels

            fig.update_layout(
                height=600,
                width=plot_width,  # Set fixed width
                margin=dict(l=150, r=150, t=50, b=30),  # Increased margins
            )

            # Convert to FigureWidget for interactivity and callbacks
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
                grid_gap="10px",
                align_items="center",
            ),
        )

        # Labels for the different metric types
        comp_metrics_label_clf = widgets.Label(
            value="Computation Metrics: ",
            layout=widgets.Layout(padding="5px 0px")
        )

        stat_metrics_label_clf = widgets.Label(
            value="Statistical Metrics: ",
            layout=widgets.Layout(padding="5px 0px")
        )

        comp_metrics_label_reg = widgets.Label(
            value="Computation Metrics: ",
            layout=widgets.Layout(padding="5px 0px")
        )

        stat_metrics_label_reg = widgets.Label(
            value="Statistical Metrics: ",
            layout=widgets.Layout(padding="5px 0px")
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
                self.metric_checkboxes["regression"]["MedAE"],
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
            )
        )

    def display(self):
        """Display the widgets and initialize the plot."""
        # Create the layout
        layout = self.create_layout()

        # Display the layout
        display(layout)

        # Initialize the plot
        self._update_plot()


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
        "_clf_datasets",
        "_reg_datasets",
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
            "MedAE": "median_absolute_error",
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
            "MedAE",
        ]

        # Extract dataset information from input data instead of filtering self
        # This prevents recursion issues
        if args and isinstance(args[0], dict):
            input_data = args[0]
            ml_task_array = np.array(input_data.get("ml_task", []))
            dataset_array = np.array(input_data.get("dataset", []))

            # Get unique datasets for classification and regression tasks
            self._clf_datasets = np.unique(
                dataset_array[ml_task_array == "classification"]
            ).tolist()
            self._reg_datasets = np.unique(
                dataset_array[ml_task_array == "regression"]
            ).tolist()
        else:
            # Fallback to empty lists if we can't determine from input
            self._clf_datasets = []
            self._reg_datasets = []

        # Create plot widget
        self._plot_widget = ParallelCoordinatePlotWidget(
            dataframe=self,
            dimension_to_column=self._dimension_to_column,
            column_to_dimension=self._column_to_dimension,
            invert_colormap=self._invert_colormap,
            clf_datasets=self._clf_datasets,
            reg_datasets=self._reg_datasets,
        )

    @property
    def _constructor(self):
        """Return the constructor for this class."""
        return MetaDataFrame

    def _repr_html_(self):
        """Display the interactive plot and controls."""
        self._plot_widget.display()
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
        self._plot_widget.update_selection()
        return self

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
        # First update the selection to ensure we have the latest state
        self.update_selection()

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

    def query(
        self, expr: str, *, inplace: bool = False, **kwargs
    ) -> pd.DataFrame | None:
        """
        Query the columns of a DataFrame with a boolean expression.

        Parameters
        ----------
        expr : str
            The query string to evaluate.

            You can refer to variables
            in the environment by prefixing them with an '@' character like
            ``@a + b``.

            You can refer to column names that are not valid Python variable names
            by surrounding them in backticks. Thus, column names containing spaces
            or punctuations (besides underscores) or starting with digits must be
            surrounded by backticks. (For example, a column named "Area (cm^2)" would
            be referenced as ```Area (cm^2)```). Column names which are Python keywords
            (like "list", "for", "import", etc) cannot be used.

            For example, if one of your columns is called ``a a`` and you want
            to sum it with ``b``, your query should be ```a a` + b``.

        inplace : bool
            Whether to modify the DataFrame rather than creating a new one.
        **kwargs
            See the documentation for :func:`eval` for complete details
            on the keyword arguments accepted by :meth:`DataFrame.query`.

        Returns
        -------
        DataFrame or None
            DataFrame resulting from the provided query expression or
            None if ``inplace=True``.

        See Also
        --------
        eval : Evaluate a string describing operations on
            DataFrame columns.
        DataFrame.eval : Evaluate a string describing operations on
            DataFrame columns.

        Notes
        -----
        The result of the evaluation of this expression is first passed to
        :attr:`DataFrame.loc` and if that fails because of a
        multidimensional key (e.g., a DataFrame) then the result will be passed
        to :meth:`DataFrame.__getitem__`.

        This method uses the top-level :func:`eval` function to
        evaluate the passed query.

        The :meth:`~pandas.DataFrame.query` method uses a slightly
        modified Python syntax by default. For example, the ``&`` and ``|``
        (bitwise) operators have the precedence of their boolean cousins,
        :keyword:`and` and :keyword:`or`. This *is* syntactically valid Python,
        however the semantics are different.

        You can change the semantics of the expression by passing the keyword
        argument ``parser='python'``. This enforces the same semantics as
        evaluation in Python space. Likewise, you can pass ``engine='python'``
        to evaluate an expression using Python itself as a backend. This is not
        recommended as it is inefficient compared to using ``numexpr`` as the
        engine.

        The :attr:`DataFrame.index` and
        :attr:`DataFrame.columns` attributes of the
        :class:`~pandas.DataFrame` instance are placed in the query namespace
        by default, which allows you to treat both the index and columns of the
        frame as a column in the frame.
        The identifier ``index`` is used for the frame index; you can also
        use the name of the index to identify it in a query. Please note that
        Python keywords may not be used as identifiers.

        For further details and examples see the ``query`` documentation in
        :ref:`indexing <indexing.query>`.

        *Backtick quoted variables*

        Backtick quoted variables are parsed as literal Python code and
        are converted internally to a Python valid identifier.
        This can lead to the following problems.

        During parsing a number of disallowed characters inside the backtick
        quoted string are replaced by strings that are allowed as a Python identifier.
        These characters include all operators in Python, the space character, the
        question mark, the exclamation mark, the dollar sign, and the euro sign.
        For other characters that fall outside the ASCII range (U+0001..U+007F)
        and those that are not further specified in PEP 3131,
        the query parser will raise an error.
        This excludes whitespace different than the space character,
        but also the hashtag (as it is used for comments) and the backtick
        itself (backtick can also not be escaped).

        In a special case, quotes that make a pair around a backtick can
        confuse the parser.
        For example, ```it's` > `that's``` will raise an error,
        as it forms a quoted string (``'s > `that'``) with a backtick inside.

        See also the Python documentation about lexical analysis
        (https://docs.python.org/3/reference/lexical_analysis.html)
        in combination with the source code in :mod:`pandas.core.computation.parsing`.

        Examples
        --------
        >>> df = pd.DataFrame({'A': range(1, 6),
        ...                    'B': range(10, 0, -2),
        ...                    'C C': range(10, 5, -1)})
        >>> df
           A   B  C C
        0  1  10   10
        1  2   8    9
        2  3   6    8
        3  4   4    7
        4  5   2    6
        >>> df.query('A > B')
           A  B  C C
        4  5  2    6

        The previous expression is equivalent to

        >>> df[df.A > df.B]
           A  B  C C
        4  5  2    6

        For columns with spaces in their name, you can use backtick quoting.

        >>> df.query('B == `C C`')
           A   B  C C
        0  1  10   10

        The previous expression is equivalent to

        >>> df[df.B == df['C C']]
           A   B  C C
        0  1  10   10
        """
        output = super().query(expr, inplace=inplace, **kwargs)
        if inplace:
            return None
        else:
            # we need to force an update of the partial dependence plot
            output._plot_widget = ParallelCoordinatePlotWidget(
                dataframe=output,
                dimension_to_column=output._dimension_to_column,
                column_to_dimension=output._column_to_dimension,
                invert_colormap=output._invert_colormap,
                clf_datasets=output._clf_datasets,
                reg_datasets=output._reg_datasets,
            )
            return output


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
df

# %%
