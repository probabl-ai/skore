"""
.. _example_basic_usage:

===========
Basic usage
===========

This example complements the :ref:`example_getting_started` example and shows
some more functionalities.
"""

# %%
# Project and UI
# --------------

# %%
import altair as alt
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import PIL

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skore import load
from skore.item import MediaItem

# %%
# Initialize a Project and launch the UI
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
import subprocess

# remove the project if it already exists
subprocess.run("rm -rf my_project_bu.skore".split())

# create the project
subprocess.run("python3 -m skore create my_project_bu".split())


# %%
from skore import load

my_project_bu = load("my_project_bu.skore")


# %%
# Storing an integer
# ^^^^^^^^^^^^^^^^^^
#
# Now, let us store our first object, for example an integer:

# %%
my_project_bu.put("my_int", 3)

# %%
# Here, the name of the object is ``my_int`` and the integer value is 3.
#
# You can read it from the project:

# %%
my_project_bu.get("my_int")

# %%
# Careful; like in a traditional Python dictionary, the ``put`` method will *overwrite* past data if you use a key which already exists!

# %%
my_project_bu.put("my_int", 30_000)

# %%
# Let us check the updated value:

# %%
my_project_bu.get("my_int")

# %%
# By using the ``delete_item`` method, you can also delete an object so that your skore UI does not become cluttered:

# %%
my_project_bu.put("my_int_2", 10)

# %%
my_project_bu.delete_item("my_int_2")

# %%
# You can display all the keys in your project:

# %%
my_project_bu.list_item_keys()

# %%
# Storing a string
# ^^^^^^^^^^^^^^^^

# %%
# We just stored a integer, now let us store some text using strings!

# %%
my_project_bu.put("my_string", "Hello world!")

# %%
my_project_bu.get("my_string")

# %%
# ``my_project_bu.get`` infers the type of the inserted object by default. For example, strings are assumed to be in Markdown format. Hence, you can customize the display of your text:

# %%
my_project_bu.put(
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

# %%
# Moreover, you can also explicitly tell `skore` the media type of an object, for example in HTML:

# %%
# Note: we use ``put_item`` instead of ``put``:
my_project_bu.put_item(
    "my_string_3",
    MediaItem.factory(
        "<p><h1>Title</h1> <b>bold</b>, <i>italic</i>, etc.</p>", media_type="text/html"
    ),
)

# %%
# Note that the media type is only used for the UI, and not in this notebook at hand:

# %%
my_project_bu.get("my_string_3")

# %%
# You can also conveniently use a Python f-string:

# %%
x = 2
y = [1, 2, 3, 4]
my_project_bu.put(
    "my_string_4", f"The value of `x` is {x} and the value of `y` is {y}."
)

# %%
# Storing many kinds of data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Python list:

# %%
my_list = [1, 2, 3, 4]
my_project_bu.put("my_list", my_list)

# %%
# Python dictionary:

# %%
my_dict = {
    "company": "probabl",
    "year": 2023,
}
my_project_bu.put("my_dict", my_dict)

# %%
# Numpy array:

# %%
my_arr = np.random.randn(3, 3)
my_project_bu.put("my_arr", my_arr)

# %%
# Pandas data frame:

# %%
my_df = pd.DataFrame(np.random.randn(3, 3))
my_project_bu.put("my_df", my_df)

# %%
# Data visualization
# ^^^^^^^^^^^^^^^^^^
#
# Note that, in the dashboard, the interactivity of plots is supported, for example for Altair and Plotly.

# %%
# Matplotlib figure:

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

my_project_bu.put("my_figure", fig)

# %%
# Altair chart:

# %%
num_points = 100
df_plot = pd.DataFrame(
    {"x": np.random.randn(num_points), "y": np.random.randn(num_points)}
)

my_altair_chart = (
    alt.Chart(df_plot)
    .mark_circle()
    .encode(x="x", y="y", tooltip=["x", "y"])
    .interactive()
    .properties(title="My title")
)

my_project_bu.put("my_altair_chart", my_altair_chart)

# %%
# .. note::
#     For Plotly figures, some users reported the following error when running Plotly cells:
#     ```
#     ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed
#     ```
#     This is a Plotly issue which is documented `here <https://github.com/plotly/plotly.py/issues/3285>`_; to solve it, we recommend installing nbformat in your environment, e.g. with
#
#     .. code-block:: bash
#
#         pip install --upgrade nbformat

# %%
# Plotly figure:

# %%
df = px.data.iris()
fig = px.scatter(
    df,
    x=df.sepal_length,
    y=df.sepal_width,
    color=df.species,
    size=df.petal_length
)

my_project_bu.put("my_plotly_fig", fig)

# %%
# Animated Plotly figure:

# %%
df = px.data.gapminder()
my_anim_plotly_fig = px.scatter(
    df,
    x="gdpPercap",
    y="lifeExp",
    animation_frame="year",
    animation_group="country",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=55,
    range_x=[100, 100000],
    range_y=[25, 90],
)

my_project_bu.put("my_anim_plotly_fig", my_anim_plotly_fig)

# %%
# PIL image:

# %%
my_pil_image = PIL.Image.new("RGB", (100, 100), color="red")
with io.BytesIO() as output:
    my_pil_image.save(output, format="png")

my_project_bu.put("my_pil_image", my_pil_image)

# %%
# Scikit-learn models and pipelines
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As skore is developed by `Probabl <https://probabl.ai>`_, the spin-off of scikit-learn, skore treats scikit-learn models and pipelines as first-class citizens.
#
# First of all, you can store a scikit-learn model:

# %%
my_model = Lasso(alpha=2)
my_project_bu.put("my_model", my_model)

# %%
# You can also store scikit-learn pipelines:

# %%
my_pipeline = Pipeline(
    [("standard_scaler", StandardScaler()), ("lasso", Lasso(alpha=2))]
)
my_project_bu.put("my_pipeline", my_pipeline)

# %%
# Moreover, you can store fitted scikit-learn pipelines:

# %%
diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
my_pipeline.fit(X, y)

my_project_bu.put("my_fitted_pipeline", my_pipeline)

# %%
# *Stay tuned for some new features!*
