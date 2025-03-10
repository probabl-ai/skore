"""
.. _example_tracking_items:

==============
Tracking items
==============

This example illustrates how skore can be used to track some items using their history,
for example tracking some ML metrics over time.
"""

# %%
# Creating and loading the skore project
# ======================================

# %%
# We create and load the skore project in the current directory:

# %%
import skore

# sphinx_gallery_start_ignore
import os
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = Path(temp_dir.name)
os.chdir(temp_dir_path)
# sphinx_gallery_end_ignore

my_project = skore.Project("my_project")

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
# We retrieve the history of the ``my_int`` item:

# %%
history = my_project.get("my_int", version="all", metadata=True)

# %%
# We can print the details of the first version of this item:

# %%

print(history[0])

# %%
# Let us construct a dataframe with the values and last updated times:

# %%
import numpy as np
import pandas as pd

list_value, list_created_at, list_updated_at = zip(
    *[(version["value"], history[0]["date"], version["date"]) for version in history]
)

df_track = pd.DataFrame(
    {
        "value": list_value,
        "created_at": list_created_at,
        "updated_at": list_updated_at,
    }
)
df_track.insert(0, "version_number", np.arange(len(df_track)))
df_track

# %%
# .. role:: python(code)
#   :language: python
#
# Notice that the ``created_at`` dates are the same for all iterations because they
# correspond to the date of the first version of the item, but the ``updated_at`` dates
# are spaced by 0.1 second (approximately) as we used :python:`time.sleep(0.1)` between
# each :func:`~skore.Project.put`.

# %%
# We can now track the value of the item over time:

# %%
import plotly.express as px

fig = px.line(
    df_track,
    x="version_number",
    y="value",
    hover_data=df_track.columns,
    markers=True,
)
fig.update_layout(xaxis_type="category")
# sphinx_gallery_start_ignore
temp_dir.cleanup()
# sphinx_gallery_end_ignore
fig

# %%
# .. note::
#   We can hover over the histories of the item to visualize the last update date for
#   example.

# %%
# Here, we focused on *how* to use skore's tracking of history of items.
# But *why* track items?
#
# * We could track some items such as machine learning scores over time to better
#   understand which feature engineering works best.
#
# * Avoid overwriting a useful metric by mistake. No results are can be lost.
#
# * The last updated time can help us reproduce an iteration of a key metric.
