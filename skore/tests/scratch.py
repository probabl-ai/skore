# %% [markdown]
#
# Generate a data for a dataframe that could be the available metadata for a given
# experiment.

# %%
import numpy as np
import pandas as pd
import plotly.graph_objects as go

rng = np.random.default_rng(42)

size = 100
index_reg_vs_clf = rng.choice([True, False], size=size)
ml_task = ["classification" if idx else "regression" for idx in index_reg_vs_clf]

r2_score = rng.uniform(0, 1, size=size)
r2_score[index_reg_vs_clf] = np.nan

accuracy_score = rng.uniform(0, 1, size=size)
precision_score = rng.uniform(0, 1, size=size)
recall_score = rng.uniform(0, 1, size=size)

accuracy_score[~index_reg_vs_clf] = np.nan
precision_score[~index_reg_vs_clf] = np.nan
recall_score[~index_reg_vs_clf] = np.nan

data = {
    "ml_task": ml_task,
    "r2_score": r2_score,
    "accuracy_score": accuracy_score,
    "precision_score": precision_score,
    "recall_score": recall_score,
}

# %%
from IPython.display import display
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

    Methods
    -------
    update_selection
        Explicitly updates the selection based on the current plot state.
    get_selection_query
        Generates a pandas query string based on the visual selection.
    to_frame
        Converts the MetaDataFrame to a standard pandas DataFrame.

    Examples
    --------
    >>> df = MetaDataFrame(data)
    >>> df  # Displays interactive parallel coordinates plot
    >>> query = df.get_selection_query()
    >>> filtered_df = df.query(query)
    """

    _metadata = [
        "_dimension_to_column",
        "_column_to_dimension",
        "_current_fig",
        "_current_dimensions",
        "_current_selection",
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
            "ML Task": "ml_task",
            "R2 Score": "r2_score",
            "Accuracy Score": "accuracy_score",
            "Precision Score": "precision_score",
            "Recall Score": "recall_score",
        }
        self._column_to_dimension = {v: k for k, v in self._dimension_to_column.items()}

        # Store the current figure, dimension filters, and selections
        self._current_fig = None
        self._current_dimensions = None
        self._current_selection = {}

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

    def _repr_html_(self):
        """
        Create and display an interactive parallel coordinates plot.

        This method is automatically called when the DataFrame is displayed
        in a Jupyter notebook. It creates a parallel coordinates plot using
        Plotly with the ML Task in the center to visualize relationships
        between different metrics.

        Returns
        -------
        str
            An empty string, as the display is handled by IPython.display.
        """
        # Rearrange columns to have ml_task in the center
        columns_order = [
            "r2_score",
            "ml_task",
            "accuracy_score",
            "precision_score",
            "recall_score",
        ]

        # Create dimensions list for parallel coordinates plot
        dimensions = []

        # Add each dimension with appropriate configuration
        for col in columns_order:
            if col == "ml_task":
                # For categorical ml_task column, set as a dimension
                values = [0 if task == "regression" else 1 for task in self["ml_task"]]
                dimensions.append(
                    dict(
                        range=[0, 1],
                        label="ML Task",
                        values=values,
                        tickvals=[0, 1],
                        ticktext=["Regression", "Classification"],
                    )
                )
            else:
                # For numerical columns, handle NaN values by setting valid range
                values = (
                    self[col].fillna(0).tolist()
                )  # Ensure proper conversion to list
                dimensions.append(
                    dict(
                        range=[0, 1],  # All metrics are between 0 and 1
                        label=self._column_to_dimension.get(
                            col, col.replace("_", " ").title()
                        ),
                        values=values,
                    )
                )

        # STEP 1: First create a regular figure (this helps ensure lines are rendered)
        regular_fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=self["ml_task"]
                    .map({"regression": 0, "classification": 1})
                    .tolist(),
                    colorscale=[[0, "blue"], [1, "red"]],
                    showscale=True,
                    colorbar=dict(
                        title="ML Task",
                        tickvals=[0, 1],
                        ticktext=["Regression", "Classification"],
                    ),
                ),
                dimensions=dimensions,
            )
        )

        # STEP 2: Convert to FigureWidget for interactivity
        fig = FigureWidget(regular_fig)

        fig.update_layout(
            title="Parallel Coordinates Plot with ML Task in Center",
            font=dict(size=14),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=600,
            width=900,
        )

        # Store the current figure and dimensions for later use
        self._current_fig = fig
        self._current_dimensions = dimensions

        # Register callbacks for interactive selections
        self._register_callbacks(fig)

        display(fig)
        return ""

    def _register_callbacks(self, fig):
        """
        Register callbacks for the interactive figure.

        This method is kept for API compatibility but currently does not
        register callbacks, as we use an explicit approach instead.

        Parameters
        ----------
        fig : plotly.graph_objects.FigureWidget
            The figure widget to register callbacks for.
        """
        pass  # We'll use an explicit approach instead of callbacks

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

        # Rest of the method remains the same
        if not self._current_selection:
            return ""

        query_parts = []

        for dim_name, range_values in self._current_selection.items():
            # Handle categorical dimension (ML Task)
            if dim_name == "ML Task":
                if range_values[0] >= 0 and range_values[1] <= 0.5:
                    query_parts.append("ml_task == 'regression'")
                elif range_values[0] > 0.5 and range_values[1] <= 1:
                    query_parts.append("ml_task == 'classification'")
                # If selection spans both categories
                elif range_values[0] <= 0.5 and range_values[1] >= 0.5:
                    # No constraint needed as it includes both categories
                    pass
            else:
                # Handle numerical dimensions
                col_name = self._dimension_to_column.get(dim_name)
                if col_name:
                    # Add range query for numerical columns
                    min_val = range_values[0]
                    max_val = range_values[1]

                    # Format with appropriate precision
                    query_parts.append(
                        f"({col_name} >= {min_val:.6f} and {col_name} <= {max_val:.6f})"
                    )

        # Join all query parts with logical AND
        if query_parts:
            return " and ".join(query_parts)
        else:
            return ""

    def to_frame(self):
        """
        Convert this MetaDataFrame to a regular pandas DataFrame.

        This method creates a standard pandas DataFrame containing the same
        data as this MetaDataFrame, without the enhanced visualization
        capabilities.

        Returns
        -------
        pandas.DataFrame
            A standard pandas DataFrame containing the same data.
        """
        return pd.DataFrame(self)


# %%
df = MetaDataFrame(data)
df

# %%
query_string = df.get_selection_query()
query_string

# %%
# %%
df.query(df.get_selection_query()).to_frame()

# %%
