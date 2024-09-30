# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting started with `skore`

# %% [markdown]
# # Introduction
#
# The purpose of this guide is to illustrate some of the main features that `skore` provides. Please refer to our instructions for installing `skore`.
#
# Given to you by [:probabl.](https://probabl.ai/), `skore` is a powerful tool that allows data scientists to create tracking and clear reports from their Python code, typically a notebook. For example, see [this HTML file](https://gist.github.com/augustebaum/6b21dbd7f7d5a584fbf2c1956692574e): download it and open it in your browser to visualize it.

# %% [markdown]
# ## Imports

# %%
# ruff: noqa

import altair as alt
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skore import load
from skore.item import MediaItem

# %% [markdown]
# # Initialize and use a Project
#
# To initialize a Project, we need to give it a name, or equivalently a file path:

# %%
# Create a project at path './project.skore'
# !python -m skore create 'project.skore'

# %% [markdown]
# This will create a Skore project directory named "project.skore" in the current directory.

# %% [markdown]
# Now that you have created the `project.skore` folder (even though nothing has yet been stored), you can run the UI (in your project root i.e. where `project.skore` is):
# ```python3
# $ python -m skore launch project.skore
# ```
#
# >*Note*: If you already had some data in your `project.skore` directory from a previous run -- you can check for that in your shell by using:
# >```python3
# >$ ls
# >```
# >and if you no longer need its objects, we recommend deleting this folder by running `rm` in your shell:
# >```python3
# >$ rm -r project.skore
# >```
# >This deletion needs to be done before the cells above: before initializing the store and before launching the UI!

# %% [markdown]
# Now that the project file exists, we can load it in our script so that we can read from and write to it:

# %%
project = load("project.skore")

# %% [markdown]
# ## Storing an integer

# %% [markdown]
# Now, let us store our first object, for example an integer:

# %%
project.put("my_int", 3)

# %% [markdown]
# Here, the name of my object is `my_int` and the integer value is 3.
#
# You can read it from the Project:

# %%
project.get("my_int")

# %% [markdown]
# Careful; like in a traditional Python dictionary, the `put` method will *overwrite* past data if you use a key which already exists!

# %%
project.put("my_int", 30_000)

# %% [markdown]
# Let us check the updated value:

# %%
project.get("my_int")

# %% [markdown]
# By using the `delete_item` method, you can also delete an object so that your `skore` UI does not become cluttered:

# %%
project.put("my_int_2", 10)

# %%
project.delete_item("my_int_2")

# %% [markdown]
# You can use `project.list_item_keys` to display all the keys in your project:

# %%
project.list_item_keys()

# %% [markdown]
# ## Storing a string

# %% [markdown]
# We just stored a integer, now let us store some text using strings!

# %%
project.put("my_string", "Hello world!")

# %%
project.get("my_string")

# %% [markdown]
# `project.get` infers the type of the inserted object by default. For example, strings are assumed to be in Markdown format. Hence, you can customize the display of your text:

# %%
project.put(
    "my_string_2",
    (
        """Hello world!, **bold**, *italic*, `code`

```python
def my_func(x):
    return x+2
```
"""
    ),
)

# %% [markdown]
# Moreover, you can also explicitly tell `skore` the media type of an object, for example in HTML:

# %%
# Note we use `put_item` instead of `put`
project.put_item(
    "my_string_3",
    MediaItem.factory(
        "<p><h1>Title</h1> <b>bold</b>, <i>italic</i>, etc.</p>", media_type="text/html"
    ),
)

# %% [markdown]
# Note that the media type is only used for the UI, and not in this notebook at hand:

# %%
project.get("my_string_3")

# %% [markdown]
# You can also conveniently use Python f-strings:

# %%
x = 2
y = [1, 2, 3, 4]
project.put("my_string_4", f"The value of `x` is {x} and the value of `y` is {y}.")

# %% [markdown]
# ## Storing many kinds of data

# %% [markdown]
# Python list:

# %%
my_list = [1, 2, 3, 4]
project.put("my_list", my_list)

# %% [markdown]
# Python dictionary:

# %%
my_dict = {
    "company": "probabl",
    "year": 2023,
}
project.put("my_dict", my_dict)

# %% [markdown]
# NumPy array:

# %%
my_arr = np.random.randn(3, 3)
project.put("my_arr", my_arr)

# %% [markdown]
# Pandas data frame:

# %%
my_df = pd.DataFrame(np.random.randn(3, 3))
project.put("my_df", my_df)

# %% [markdown]
# ## Data visualization

# %% [markdown]
# Matplotlib Figures:

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
# Altair Charts:

# %%
num_points = 100
df_plot = pd.DataFrame(
    {"x": np.random.randn(num_points), "y": np.random.randn(num_points)}
)

my_chart = (
    alt.Chart(df_plot)
    .mark_circle()
    .encode(x="x", y="y", tooltip=["x", "y"])
    .interactive()
    .properties(title="My title")
)

project.put("my_chart", my_chart)

# %% [markdown]
# PIL images:

# %%
pil_image = PIL.Image.new("RGB", (100, 100), color="red")
with io.BytesIO() as output:
    pil_image.save(output, format="png")

project.put("pil_image", pil_image)

# %% [markdown]
# ## Scikit-learn models and pipelines
#
# As `skore` is developed by :probabl., the spin-off of scikit-learn, `skore` treats scikit-learn models and pipelines as first-class citizens.
#
# First of all, you can store a scikit-learn model:

# %%
my_model = Lasso(alpha=2)
project.put("my_model", my_model)

# %% [markdown]
# You can also store scikit-learn pipelines:

# %%
my_pipeline = Pipeline(
    [("standard_scaler", StandardScaler()), ("lasso", Lasso(alpha=2))]
)
project.put("my_pipeline", my_pipeline)

# %% [markdown]
# Moreover, you can store fitted scikit-learn pipelines:

# %%
diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
my_pipeline.fit(X, y)

project.put("my_fitted_pipeline", my_pipeline)

# %% [markdown]
# _Stay tuned for some new features!_

# %% [markdown]
# ---
# # Manipulating the skore UI
#
# The following is just some `skore` strings that we generate in order to provide more context on the obtained report.

# %%
project.put_item(
    "my_comment_1",
    MediaItem.factory(
        "<p><h1>Welcome to skore!</h1>Given to you by :probabl., skore is a powerful tool that allows data scientists to create tracking and clear reports from their Python code, typically a notebook. This HTML document is actually a skore report generated using the `basic_usage.ipynb` notebook that has been exported (into HTML)!<p>",
        media_type="text/html",
    ),
)

# %%
project.put_item(
    "my_comment_2",
    MediaItem.factory("<p><h2>Integers</h1></p>", media_type="text/html"),
)

# %%
project.put_item(
    "my_comment_3", MediaItem.factory("<p><h2>Strings</h1></p>", media_type="text/html")
)

# %%
project.put_item(
    "my_comment_4",
    MediaItem.factory("<p><h2>Many kinds of data</h1></p>", media_type="text/html"),
)

# %%
project.put_item(
    "my_comment_5",
    MediaItem.factory("<p><h2>Altair plots</h1></p>", media_type="text/html"),
)

# %%
project.put_item(
    "my_comment_6",
    MediaItem.factory(
        "<p><h2>Scikit-learn models and pipelines</h1></p>", media_type="text/html"
    ),
)

# %%
project.put_item(
    "my_comment_7",
    MediaItem.factory(
        "<p><h2>Manipulating the skore report</h1></p>", media_type="text/html"
    ),
)

# %%
project.put_item(
    "my_comment_8",
    MediaItem.factory(
        "<p>Once you have created cells in your Skore report, you can place them where you want: move them up or down, delete them, for example put one to the left of a graph to comment on it.</p>",
        media_type="text/html",
    ),
)

# %%
project.put("my_chart_2", my_chart)
