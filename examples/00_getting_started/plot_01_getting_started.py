"""
.. _example_getting_started:

==========================
Getting started with skore
==========================
"""

# %%
# This getting started guide illustrates how to use skore and why:
#
# #. Track and visualize your ML/DS results using skore's :class:`~skore.Project` and UI.
#
# #. Machine learning diagnostics: get assistance when developing your ML/DS projects.
#
#    * Scikit-learn compatible :class:`skore.CrossValidationReporter` and
#      :func:`skore.train_test_split` provide insights and checks on cross-validation
#      and train-test-split.

# %%
# Track and visualize: skore project and UI
# =========================================
#
# A key feature of skore is its :class:`~skore.Project` that allows to store
# items of many types then visualize them in a dashboard called the skore UI.

# %%
# Setup: creating and loading a skore project, and launching the skore UI
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# .. note::
#
#   If we do not wish for our skore project to be stored in a *temporary* folder, we
#   can simply create and load the project in the current directory with:
#
#   .. code-block:: python
#
#     import skore
#
#     my_project = skore.create("my_project")
#
#   This would create a skore project directory named ``my_project.skore`` in our
#   current directory.

# %%
# Here, we start by creating a temporary directory to store our project so that we can
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
# Then, *from our shell* (in the same directory), we can start the UI locally:
#
# .. code-block:: bash
#
#     $ skore launch "my_project"
#
# This will automatically open a browser at the UI's location.
#
# Now that the project exists, we can write some Python code (in the same
# directory) to add (:func:`~skore.Project.put`) some useful items in it,
# with a key-value convention:

# %%
my_project.put("my_int", 3)

# %%
# We can retrieve the value of an item:

# %%
my_project.get("my_int")

# %%
# Skore project and storage: example of hyperparameter sweep
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As an illustration of the usage of the skore project and storage with a machine
# learning motivation, let us perform a hyperparameter sweep and store relevant
# information in the skore project.

# %%
# We search for the ``alpha`` hyperparameter of a Ridge regression on the
# Diabetes dataset:

# %%
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

X, y = load_diabetes(return_X_y=True)

gs_cv = GridSearchCV(
    Ridge(),
    param_grid={"alpha": np.logspace(-3, 5, 50)},
    scoring="neg_root_mean_squared_error",
)
gs_cv.fit(X, y)

# %%
# Now, we store the hyperparameter's metrics in a dataframe and make a custom
# plot:

# %%
import pandas as pd

df = pd.DataFrame(gs_cv.cv_results_)
df.insert(len(df.columns), "rmse", -df["mean_test_score"].values)
df[["param_alpha", "rmse"]].head()

# %%
import matplotlib.pyplot as plt

fig = plt.figure(layout="constrained", dpi=200)
plt.plot(df["param_alpha"], df["rmse"])
plt.xscale("log")
plt.xlabel("Alpha hyperparameter")
plt.ylabel("RMSE")
plt.title("Ridge regression")
plt.show()

# %%
# |
# Finally, we store some relevant information to our skore project, so that we
# can visualize them later in the skore UI for example:

# %%
my_project.put("my_gs_cv", gs_cv)
my_project.put("my_df", df)
my_project.put("my_fig", fig)

# %%
# .. seealso::
#
#   For more information about the functionalities and the different types
#   of items that we can store in a skore :class:`~skore.Project`,
#   see :ref:`example_overview_skore_project`.

# %%
# Manipulating the skore UI
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The skore UI is a very efficient tool to track and visualize the items in your
# project, such as grid search or cross-validation results.
#
# #.  On the top menu, by default, you can observe that you are in a *View* called
#     ``default``. You can rename this view or create another one.
#
# #.  From the *Items* section on the left, you can add stored items to this
#     view, either by clicking on ``+`` or by dragging an item to the right.
#
# #.  In the skore UI on the right, you can drag-and-drop items to re-order them,
#     remove items, etc.
#
# .. image:: https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/2024_10_31_skore_demo_compressed.gif
#   :alt: Getting started with ``skore`` demo

# %%
# Tracking the history of items
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Skore does not overwrite items with the same name, instead it stores their history so
# that, from the skore UI, we could visualize their different histories:
#
# .. image:: https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/2024_11_21_tracking_comp.gif
#   :alt: Tracking the history of an item from the skore UI
#
# This functionality is very useful to never lose some key machine learning metrics,
# for example.
#
# .. seealso::
#
#   For more information about the tracking of items using their history,
#   see :ref:`example_historization_items`.

# %%
# Machine learning diagnostics: enhancing scikit-learn functions
# ==============================================================
#
# Skore wraps some key scikit-learn functions to automatically provide
# diagnostics and checks when using them, as a way to facilitate good practices.

# %%
# Cross-validation with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In order to assist its users when programming, skore has implemented a
# :class:`skore.CrossValidationReporter` function that wraps scikit-learn's
# :func:`sklearn.model_selection.cross_validate`.
#
# On the same previous data and a Ridge regressor (with default ``alpha`` value),
# let us create a CrossValidationReporter.

# %%
from skore import CrossValidationReporter

reporter = CrossValidationReporter(Ridge(), X, y, cv=5)
reporter.plot

# %%
# Hence, we can observe some key metrics and get insights on our cross-validation.

# %%
# .. seealso::
#
#   For more information about the motivation and usage of :class:`skore.CrossValidationReporter`,
#   see :ref:`example_cross_validate`.

# %%
# Train-test split with skore
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Skore has implemented a :func:`skore.train_test_split` function that wraps
# scikit-learn's :func:`sklearn.model_selection.train_test_split`.
#
# For example, it can raise warnings when there is class imbalance in the data to
# provide methodological advice:

# %%
X = np.arange(400).reshape((200, 2))
y = [0] * 150 + [1] * 50

X_train, X_test, y_train, y_test = skore.train_test_split(
    X=X, y=y, test_size=0.2, random_state=0
)

# %%
# .. seealso::
#
#   For more information about the motivation and usage of
#   :func:`skore.train_test_split`, see :ref:`example_train_test_split`.

# %%
# .. admonition:: Stay tuned for some new features!
#
#    Feel free to join our `Discord <https://discord.gg/scBZerAGwW>`_!

# %%
# Cleanup the project
# -------------------
#
# Removing the temporary directory:

# %%
temp_dir.cleanup()
