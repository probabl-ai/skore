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
# This guide showcases the features of `skrub`, an open-source package that aims at enable data scientist to:
# 1. Store objects of different types from their Python code: from python lists to scikit-learn fitted pipelines and plotly figures.
# 2. They can track and visualize these stored objects on a dashboard.
# 3. This dashboard can be exported into a HTML file.

# %% [markdown]
# ## Imports

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skore import load

# %% [markdown]
# ## Initialize a Project and launch the UI
#
# Initialize a `skore` Project from your shell:
# ```bash
# $ python -m skore create 'project.skore'
# ```
# This will create a Skore project directory named `project.skore` in the current directory.
#
# Then, you can run the UI (in your project root i.e. where `project.skore` is) from your shell:
# ```bash
# $ python -m skore launch project.skore
# ```
#
# Now that the project file exists, we can load it in our notebook so that we can read from and write to it:

# %%
project = load("project.skore")

# %% [markdown]
# ## Storing an integer

# %% [markdown]
# Storing an integer:

# %%
project.put("my_int", 3)

# %% [markdown]
# Here, the name of my stored item is `my_int` and the integer value is 3.

# %% [markdown]
# For a `pandas` data frame:

# %%
my_df = pd.DataFrame(np.random.randn(3, 3))

project.put("my_df", my_df)

# %% [markdown]
# for a `matplotlib` figure:

# %%
x = np.linspace(0, 2, 100)

fig, ax = plt.subplots(figsize=(5, 2.7), layout="constrained")
ax.plot(x, x, label="linear")
ax.plot(x, x**2, label="quadratic")
ax.plot(x, x**3, label="cubic")
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_title("Simple Plot")
ax.legend()
plt.show()

project.put("my_figure", fig)

# %% [markdown]
# For a `scikit-learn` fitted pipeline:

# %%
diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]

my_pipeline = Pipeline(
    [("standard_scaler", StandardScaler()), ("lasso", Lasso(alpha=2))]
)
my_pipeline.fit(X, y)

project.put("my_fitted_pipeline", my_pipeline)
