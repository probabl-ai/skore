"""
=================================
1) Getting started with ``skore``
=================================

This example runs the :ref:`getting_started` guide.

``skore`` UI
------------

This section provides a quick start to the ``skore`` UI, an open-source package that aims to enable data scientists to:

#. Store objects of different types from their Python code: python lists, ``scikit-learn`` fitted pipelines, ``plotly`` figures, and more.
#. Track and visualize these stored objects on a user-friendly dashboard.
#. Export the dashboard to a HTML file.

Initialize a Project and launch the UI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From your shell, initialize a skore project, here named ``my_project_gs``, that
will be in your current working directory:
"""

# %%
import subprocess

# remove the project if it already exists
subprocess.run("rm -rf my_project_gs.skore".split())

# create the project
subprocess.run("python3 -m skore create my_project_gs".split())

# %%
# This will create a ``skore`` project directory named ``my_project_gs`` in the
# current directory.
#
# From your shell (in the same directory), start the UI locally:
#
# .. code:: console
#
#     python -m skore launch "my_project_gs"
#
# This will automatically open a browser at the UI's location.
#
# Now that the project file exists, we can load it in our notebook so that we can
# read from and write to it:

# %%
from skore import load

my_project_gs = load("my_project_gs.skore")

# %%
# Storing some items
# ^^^^^^^^^^^^^^^^^^
#
# Storing an integer:

# %%
my_project_gs.put("my_int", 3)

# %%
# Here, the name of my stored item is ``my_int`` and the integer value is 3.

# %%
my_project_gs.get("my_int")

# %%
# For a ``pandas`` data frame:

# %%
import numpy as np
import pandas as pd

my_df = pd.DataFrame(np.random.randn(3, 3))

my_project_gs.put("my_df", my_df)

# %%
my_project_gs.get("my_df")

# %%
# For a ``matplotlib`` figure:

# %%
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
_ = ax.plot(x)

my_project_gs.put("my_figure", fig)

# %%
# For a ``scikit-learn`` fitted pipeline:

# %%
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
my_pipeline = Pipeline(
    [("standard_scaler", StandardScaler()), ("lasso", Lasso(alpha=2))]
)
my_pipeline.fit(X, y)

my_project_gs.put("my_fitted_pipeline", my_pipeline)

# %%
my_project_gs.get("my_fitted_pipeline")

# %%
# Back to the dashboard
# ^^^^^^^^^^^^^^^^^^^^^
#
# #. On the top left, create a new ``View``.
# #. From the ``Elements`` section on the bottom left, you can add stored items to this view, either by double-cliking on them or by doing drag-and-drop.
#
# .. image:: https://raw.githubusercontent.com/sylvaincom/sylvaincom.github.io/master/files/probabl/skore/2024_10_14_skore_demo.gif
#    :alt: Getting started with ``skore`` demo
