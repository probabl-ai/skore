"""
.. _example_historization_items:

==================================
Tracking items using their history
==================================

This example illustrates how skore can be used to track some items using their history,
for example tracking some ML metrics over time.
"""

# %%
# Creating and loading the skore project
# ======================================

# %%
# We start by creating a temporary directory to store our project such that we can
# easily clean it after executing this example. If you want to keep the project,
# you have to skip this section.
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

# %%
import subprocess

# create the skore project
subprocess.run(
    f"python3 -m skore create my_project_track --working-dir {temp_dir.name}".split()
)

# %%
from skore import load

my_project_track = load(temp_dir_path / "my_project_track.skore")

# %%
# Tracking an integer
# ===================

# %%
# Let us store several integer values for a same item called ``my_int``, each storage
# being separated by 0.1 second:

# %%
import time

my_project_track.put("my_int", 4)
time.sleep(0.1)
my_project_track.put("my_int", 9)
time.sleep(0.1)
my_project_track.put("my_int", 16)

# %%
# Let us retrieve the history of this item:

# %%
item_histories = my_project_track.get_item_versions("my_int")

# %%
# Let us print the first history (first iteration) of this item:

# %%
item_history = item_histories[0]
print(item_history)
print(item_history.primitive)
print(item_history.created_at)
print(item_history.updated_at)

# %%
# Same, but for the second iteration:

# %%
item_history = item_histories[1]
print(item_history.primitive)
print(item_history.created_at)
print(item_history.updated_at)

# %%
# Let us construct a dataframe with the values and last updated times:

# %%
import numpy as np
import pandas as pd

list_primitive, list_created_at, list_updated_at = zip(
    *[(elem.primitive, elem.created_at, elem.updated_at) for elem in item_histories]
)

df_track = pd.DataFrame(
    {
        "primitive": list_primitive,
        "created_at": list_created_at,
        "updated_at": list_updated_at,
    }
)
df_track.insert(0, "iteration_number", np.arange(len(df_track)))
df_track

# %%
# Tracking the value of the item over time:

# %%
import plotly.express as px

fig = px.line(
    df_track,
    x="iteration_number",
    y="primitive",
    hover_data=df_track.columns,
    markers=True,
)
fig.update_layout(xaxis_type="category")
fig

# %%
# Here, wo focused on `how` to use skore's tracking of history of items.
# `Why` track items? For example, we could track some machine learning
# scores over time to understand better which feature engineering works best.
# In the following, we explore skore's :func:`skore.cross_validate` that natively
# includes tracking.

# %%
# .. _example_track_cv:
# Tracking the results of several runs of :func:`skore.cross_validate`
# ====================================================================

# %%
# In the :ref:`example_cross_validate` example, we saw why and how to use our
# :func:`skore.cross_validate` function.
# Now, let us see how we can use the tracking of items with this function.

# %%
# Let us run several cross-validations:

# %%
from sklearn import datasets
from sklearn.linear_model import Lasso
import skore

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = Lasso()

for alpha in [0.5, 1, 2]:
    cv_results = skore.cross_validate(
        Lasso(alpha=alpha), X, y, cv=5, project=my_project_track
    )

# %%
# We can compare the metrics of each run of the cross-validation (on all splits):

# %%
fig_plotly = my_project_track.get_item("cross_validation_aggregated").plot
fig_plotly
