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

df = pd.DataFrame(data)

# %%
df

# %%

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
        dimensions.append(
            dict(
                range=[0, 1],
                label="ML Task",
                values=[0 if task == "regression" else 1 for task in df["ml_task"]],
                tickvals=[0, 1],
                ticktext=["Regression", "Classification"],
            )
        )
    else:
        # For numerical columns, handle NaN values by setting valid range
        values = df[col].copy()
        dimensions.append(
            dict(
                range=[0, 1],  # All metrics are between 0 and 1
                label=col.replace("_", " ").title(),
                values=values,
            )
        )

# Create the parallel coordinates plot
fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=df["ml_task"].map({"regression": 0, "classification": 1}),
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

# Update layout
fig.update_layout(
    title="Parallel Coordinates Plot with ML Task in Center",
    font=dict(size=14),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=600,
    width=800,
)

# Show the plot
fig.show()

# %%
from IPython.display import display


class MetaDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return MetaDataFrame

    def _repr_html_(self):
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
                dimensions.append(
                    dict(
                        range=[0, 1],
                        label="ML Task",
                        values=[
                            0 if task == "regression" else 1 for task in df["ml_task"]
                        ],
                        tickvals=[0, 1],
                        ticktext=["Regression", "Classification"],
                    )
                )
            else:
                # For numerical columns, handle NaN values by setting valid range
                values = df[col].copy()
                dimensions.append(
                    dict(
                        range=[0, 1],  # All metrics are between 0 and 1
                        label=col.replace("_", " ").title(),
                        values=values,
                    )
                )

        # Create the parallel coordinates plot
        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=df["ml_task"].map({"regression": 0, "classification": 1}),
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

        fig.update_layout(
            title="Parallel Coordinates Plot with ML Task in Center",
            font=dict(size=14),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=600,
            width=900,
        )

        display(fig)

        return ""


# %%
df = MetaDataFrame(df)
df

# %%
