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
# We start by creating a temporary directory to store our project so that we can
# easily clean it after executing this example:

# %%
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

# %%
# We create and load the skore project from this temporary directory:

# %%
import skore

my_project = skore.create("my_project", working_dir=temp_dir_path)

# %%
# Tracking an integer
# ===================

# %%
# Let us store several integer values for a same item called ``my_int``, each storage
# being separated by 0.1 second:

# %%
import time

my_project.put("my_int", 4)
time.sleep(0.1)
my_project.put("my_int", 9)
time.sleep(0.1)
my_project.put("my_int", 16)

# %%
# .. note::
#
#   We could launch the skore dashboard with:
#
#   .. code-block:: bash
#
#       skore launch "my_project"
#
#   and, from the skore UI, we could visualize the different histories of the ``my_int``
#   item:
#
#   .. image:: https://raw.githubusercontent.com/sylvaincom/sylvaincom.github.io/master/files/probabl/skore/2024_11_21_tracking_comp.gif
#       :alt: Tracking the history of an item from the skore UI

# %%
# We retrieve the history of the ``my_int`` item:

# %%
item_histories = my_project.get_item_versions("my_int")

# %%
# We can print the first history (first iteration) of this item:

# %%
item_history = item_histories[0]
print(item_history)
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
# .. role:: python(code)
#   :language: python
#
# Notice that the ``created_at`` dates are the same for all iterations because they
# correspond to the same item, but the ``updated_at`` dates are spaced by 0.1 second
# (approximately) as we used :python:`time.sleep(0.1)` between each
# :func:`~skore.Project.put`.

# %%
# We can now track the value of the item over time:

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
# .. note::
#   We can hover over the histories of the item to visualize the last update date for
#   example.

# %%
# Here, we focused on `how` to use skore's tracking of history of items.
# But `why` track items?
#
# * We could track some items such as machine learning scores over time to better
#   understand which feature engineering works best.
#
# * Avoid overwriting a useful metric by mistake. No results are can be lost.
#
# * The last updated time can help us reproduce an iteration of a key metric.
#
# In the following, we explore skore's :func:`skore.cross_validate` that natively
# includes tracking.

# %%
# .. _example_track_cv:
# Tracking the results of several runs of :func:`skore.cross_validate`
# ====================================================================

# %%
# The :ref:`example_cross_validate` example explains why and how to use the
# :func:`skore.cross_validate` function.
# Here, let us see how we can use the tracking of items with this function.

# %%
# We run several cross-validations using several values of a hyperparameter:

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
        Lasso(alpha=alpha), X, y, cv=5, project=my_project
    )

# %%
# We can compare the metrics of each run of the cross-validation (on all splits):

# %%
fig_plotly = my_project.get_item("cross_validation_aggregated").plot
fig_plotly

# %%
# Hence, we can observe that the first run, with ``alpha=0.5``, works better.

# %%
# Cleanup the project
# -------------------
#
# Removing the temporary directory:

# %%
temp_dir.cleanup()
