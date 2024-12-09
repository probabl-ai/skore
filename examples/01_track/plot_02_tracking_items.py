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
#   .. image:: https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/2024_11_21_tracking_comp.gif
#       :alt: Tracking the history of an item from the skore UI
#
#   There is also an activity feed functionality:
#
#   .. image:: https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/2024_12_09_skore_activity_feed.png
#       :alt: Activity feed on the skore UI

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
# In the following, we explore skore's :class:`skore.CrossValidationReporter` that natively
# includes tracking.

# %%
# .. _example_track_cv:
# Tracking the results of several runs of :class:`skore.CrossValidationReporter`
# ==============================================================================

# %%
# The :ref:`example_cross_validate` example explains why and how to use the
# :class:`skore.CrossValidationReporter` class.
# Here, let us see how we can use the tracking of items with this function.

# %%
# Let us load some data:

# %%
from sklearn import datasets

X, y = datasets.load_diabetes(return_X_y=True)
X, y = X[:150], y[:150]

# %%
# Suppose that some users are coding in their draft notebook and are iterating on some
# cross-validations in separate cells and forgot to store the intermediate results:

# %%
import skore
from sklearn.linear_model import Lasso

reporter = skore.CrossValidationReporter(Lasso(alpha=0.5), X, y, cv=5)
my_project.put("cross_validation", reporter)

# %%
reporter = skore.CrossValidationReporter(Lasso(alpha=1), X, y, cv=5)
my_project.put("cross_validation", reporter)

# %%
reporter = skore.CrossValidationReporter(Lasso(alpha=2), X, y, cv=5)
my_project.put("cross_validation", reporter)

# %%
# Thanks to the storage in skore, we can compare the metrics of each
# cross-validation run (on all splits):

# %%
from skore.item.cross_validation_aggregation_item import CrossValidationAggregationItem

aggregated_cv_results = CrossValidationAggregationItem.factory(
    my_project.get_item_versions("cross_validation")
)

fig_plotly = aggregated_cv_results.plot
fig_plotly

# %%
# Hence, we can observe that the first run, with ``alpha=0.5``, works better.

# %%
# .. note::
#   The good practice, instead of running several cross-validations with different
#   values of the hyperparameter, would have been to use a
#   :func:`sklearn.model_selection.GridSearchCV`.

# %%
# Cleanup the project
# -------------------
#
# Removing the temporary directory:

# %%
temp_dir.cleanup()
