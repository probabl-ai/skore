# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting started with `skore`
#
# This guide provides a quick start to `skore`, an open-source package that aims at enable data scientist to:
# 1. Store objects of different types from their Python code: python lists, `scikit-learn` fitted pipelines, `plotly` figures, and more.
# 2. **Track** and  **visualize** these stored objects on a user-friendly dashboard.
# 3. Export the dashboard to a HTML file.

# %% [markdown]
# ## Initialize a Project and launch the UI
#
# From your shell, initialize a `skore` project, here named `project.skore`, that will be in your current working directory:
# ```bash
# python -m skore create "project.skore"
# ```
# This will create a skore project directory named `project.skore` in the current directory.
#
# From your shell (in the same directory), start the UI locally:
# ```bash
# python -m skore launch "project.skore"
# ```
# This will automatically open a browser at the UI's location.
#
# Now that the project file exists, we can load it in our notebook so that we can read from and write to it:

# %%
from skore import load

project = load("project.skore")

# %% [markdown]
# ## Storing some items

# %% [markdown]
# Storing an integer:

# %%
project.put("my_int", 3)

# %% [markdown]
# Here, the name of my stored item is `my_int` and the integer value is 3.

# %% [markdown]
# For a `pandas` data frame:

# %%
import numpy as np
import pandas as pd

my_df = pd.DataFrame(np.random.randn(3, 3))
project.put("my_df", my_df)

# %% [markdown]
# for a `matplotlib` figure:

# %%
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
ax.plot(x)
project.put("my_figure", fig)

# %% [markdown]
# For a `scikit-learn` fitted pipeline:

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
project.put("my_fitted_pipeline", my_pipeline)

# %% [markdown]
# ## Back to the dashboard
#
# 1. On the top left, create a new `View`.
# 2. From the `Elements` section on the bottom left, you can add stored items to this view, either by double-cliking on them or by doing drag-and-drop.

# %% [markdown]
#
