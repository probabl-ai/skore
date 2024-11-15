"""
.. _example_overview_skore_ui:

========================
Overview of the skore UI
========================

This example provides an overview of the functionalities and the different types
of items that you can store in a skore :class:`~skore.Project`.
"""

# %%
# Creating and loading a skore project
"""
.. _example_overview_skore_ui:

========================
Overview of the skore UI
========================

This example provides an overview of the functionalities and the different types
of items that you can store in a skore :class:`~skore.Project`.
"""
import tempfile
from pathlib import Path

import skore

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

my_project_ui = skore.create("my_project_ui.skore", working_dir=temp_dir_path)


# %%
# Storing integers
# ================
#
# Now, let us store our first object using :func:`~skore.Project.put`, for example an
# integer:

# %%
my_project_ui.put("my_int", 3)

# %%
# Here, the name of the object is ``my_int`` and the integer value is 3.
#
# You can read it from the project by using :func:`~skore.Project.get`:

# %%
my_project_ui.get("my_int")

# %%
# Careful; like in a traditional Python dictionary, the ``put`` method will *overwrite*
# past data if you use a key which already exists!

# %%
my_project_ui.put("my_int", 30_000)

# %%
# Let us check the updated value:

# %%
my_project_ui.get("my_int")

# %%
# By using the :func:`~skore.Project.delete_item` method, you can also delete an object
# so that your skore UI does not become cluttered:

# %%
my_project_ui.put("my_int_2", 10)

# %%
my_project_ui.delete_item("my_int_2")

# %%
# You can display all the keys in your project:

# %%
my_project_ui.list_item_keys()

# %%
# Storing strings and texts
# =========================

# %%
# We just stored a integer, now let us store some text using strings!

# %%
my_project_ui.put("my_string", "Hello world!")

# %%
my_project_ui.get("my_string")

# %%
# :func:`~skore.Project.get` infers the type of the inserted object by default. For
# example, strings are assumed to be in Markdown format. Hence, you can customize the
# display of your text:

# %%
my_project_ui.put(
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
# Moreover, you can also explicitly tell skore the media type of an object, for example
# in HTML:

# %%
from skore.item import MediaItem

my_project_ui.put_item(
    "my_string_3",
    MediaItem.factory(
        "<p><h1>Title</h1> <b>bold</b>, <i>italic</i>, etc.</p>", media_type="text/html"
    ),
)

# %%
# .. note::
#   We used :func:`~skore.Project.put_item` instead of :func:`~skore.Project.put`.

# %%
# Note that the media type is only used for the UI, and not in this notebook at hand:

# %%
my_project_ui.get("my_string_3")

# %%
# You can also conveniently use a Python f-string:

# %%
x = 2
y = [1, 2, 3, 4]
my_project_ui.put(
    "my_string_4", f"The value of `x` is {x} and the value of `y` is {y}."
)

# %%
# Storing many kinds of data
# ==========================

# %%
# Python list:

# %%
my_list = [1, 2, 3, 4]
my_project_ui.put("my_list", my_list)
my_list

# %%
# Python dictionary:

# %%
my_dict = {
    "company": "probabl",
    "year": 2023,
}
my_project_ui.put("my_dict", my_dict)
my_dict

# %%
# Numpy array:

# %%
import numpy as np

my_arr = np.random.randn(3, 3)
my_project_ui.put("my_arr", my_arr)
my_arr

# %%
# Pandas data frame:

# %%
import pandas as pd

my_df = pd.DataFrame(np.random.randn(10, 5))
my_project_ui.put("my_df", my_df)
my_df.head()

# %%
# Storing data visualizations
# ===========================
#
# Note that, in the dashboard, the interactivity of plots is supported, for example for
# Altair and Plotly.

# %%
# Matplotlib figure:

# %%
import plotly as plt

x = np.linspace(0, 2, 100)

fig, ax = plt.subplots(layout="constrained", dpi=200)
ax.plot(x, x, label="linear")
ax.plot(x, x**2, label="quadratic")
ax.plot(x, x**3, label="cubic")
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_title("Simple Plot")
ax.legend()
plt.show()

my_project_ui.put("my_figure", fig)

# %%
# |
# Altair chart:

# %%
import altair as alt

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

my_project_ui.put("my_altair_chart", my_altair_chart)

# %%
# .. note::
#     For Plotly figures, some users reported the following error when running Plotly
#     cells: ``ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not
#     installed``. This is a Plotly issue which is documented `here
#     <https://github.com/plotly/plotly.py/issues/3285>`_; to solve it, we recommend
#     installing nbformat in your environment, e.g. with:
#
#     .. code-block:: console
#
#         pip install --upgrade nbformat

# %%
# Plotly figure:

# %%
import plotly.express as px

df = px.data.iris()
fig = px.scatter(
    df, x=df.sepal_length, y=df.sepal_width, color=df.species, size=df.petal_length
)

my_project_ui.put("my_plotly_fig", fig)

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
    range_x=[100, 100_000],
    range_y=[25, 90],
)

my_project_ui.put("my_anim_plotly_fig", my_anim_plotly_fig)

# %%
# PIL image:

# %%
import io

import PIL

my_pil_image = PIL.Image.new("RGB", (100, 100), color="red")
with io.BytesIO() as output:
    my_pil_image.save(output, format="png")

my_project_ui.put("my_pil_image", my_pil_image)

# %%
# Storing scikit-learn models and pipelines
# =========================================
#
# As skore is developed by `Probabl <https://probabl.ai>`_, the spin-off of
# scikit-learn, skore treats scikit-learn models and pipelines as first-class citizens.
#
# First of all, you can store a scikit-learn model:

# %%
from sklearn.linear_model import Lasso

my_model = Lasso(alpha=2)
my_project_ui.put("my_model", my_model)
my_model

# %%
# You can also store scikit-learn pipelines:

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline(
    [("standard_scaler", StandardScaler()), ("lasso", Lasso(alpha=2))]
)
my_project_ui.put("my_pipeline", my_pipeline)
my_pipeline

# %%
# Moreover, you can store fitted scikit-learn pipelines:

# %%
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
my_pipeline.fit(X, y)

my_project_ui.put("my_fitted_pipeline", my_pipeline)
my_pipeline

# %%
# Cleanup the project
# -------------------
#
# Remove the temporary directory:
temp_dir.cleanup()

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import skore
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_project_ui = skore.create("my_project_ui.skore", working_dir=temp_dir_path)


# %%
# Storing integers
# ================
#
# Now, let us store our first object using :func:`~skore.Project.put`, for example an
# integer:

# %%
my_project_ui.put("my_int", 3)

# %%
# Here, the name of the object is ``my_int`` and the integer value is 3.
#
# You can read it from the project by using :func:`~skore.Project.get`:

# %%
my_project_ui.get("my_int")

# %%
# Careful; like in a traditional Python dictionary, the ``put`` method will *overwrite*
# past data if you use a key which already exists!

# %%
my_project_ui.put("my_int", 30_000)

# %%
# Let us check the updated value:

# %%
my_project_ui.get("my_int")

# %%
# By using the :func:`~skore.Project.delete_item` method, you can also delete an object
# so that your skore UI does not become cluttered:

# %%
my_project_ui.put("my_int_2", 10)

# %%
my_project_ui.delete_item("my_int_2")

# %%
# You can display all the keys in your project:

# %%
my_project_ui.list_item_keys()

# %%
# Storing strings and texts
# =========================

# %%
# We just stored a integer, now let us store some text using strings!

# %%
my_project_ui.put("my_string", "Hello world!")

# %%
my_project_ui.get("my_string")

# %%
# :func:`~skore.Project.get` infers the type of the inserted object by default. For
# example, strings are assumed to be in Markdown format. Hence, you can customize the
# display of your text:

# %%
my_project_ui.put(
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
# Moreover, you can also explicitly tell skore the media type of an object, for example
# in HTML:

# %%

my_project_ui.put_item(
    "my_string_3",
    MediaItem.factory(
        "<p><h1>Title</h1> <b>bold</b>, <i>italic</i>, etc.</p>", media_type="text/html"
    ),
)

# %%
# .. note::
#   We used :func:`~skore.Project.put_item` instead of :func:`~skore.Project.put`.

# %%
# Note that the media type is only used for the UI, and not in this notebook at hand:

# %%
my_project_ui.get("my_string_3")

# %%
# You can also conveniently use a Python f-string:

# %%
x = 2
y = [1, 2, 3, 4]
my_project_ui.put(
    "my_string_4", f"The value of `x` is {x} and the value of `y` is {y}."
)

# %%
# Storing many kinds of data
# ==========================

# %%
# Python list:

# %%
my_list = [1, 2, 3, 4]
my_project_ui.put("my_list", my_list)
my_list

# %%
# Python dictionary:

# %%
my_dict = {
    "company": "probabl",
    "year": 2023,
}
my_project_ui.put("my_dict", my_dict)
my_dict

# %%
# Numpy array:

# %%

my_arr = np.random.randn(3, 3)
my_project_ui.put("my_arr", my_arr)
my_arr

# %%
# Pandas data frame:

# %%

my_df = pd.DataFrame(np.random.randn(10, 5))
my_project_ui.put("my_df", my_df)
my_df.head()

# %%
# Storing data visualizations
# ===========================
#
# Note that, in the dashboard, the interactivity of plots is supported, for example for
# Altair and Plotly.

# %%
# Matplotlib figure:

# %%

x = np.linspace(0, 2, 100)

fig, ax = plt.subplots(layout="constrained", dpi=200)
ax.plot(x, x, label="linear")
ax.plot(x, x**2, label="quadratic")
ax.plot(x, x**3, label="cubic")
ax.set_xlabel("x label")
ax.set_ylabel("y label")
ax.set_title("Simple Plot")
ax.legend()
plt.show()

my_project_ui.put("my_figure", fig)

# %%
# |
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

my_project_ui.put("my_altair_chart", my_altair_chart)

# %%
# .. note::
#     For Plotly figures, some users reported the following error when running Plotly
#     cells: ``ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not
#     installed``. This is a Plotly issue which is documented `here
#     <https://github.com/plotly/plotly.py/issues/3285>`_; to solve it, we recommend
#     installing nbformat in your environment, e.g. with:
#
#     .. code-block:: console
#
#         pip install --upgrade nbformat

# %%
# Plotly figure:

# %%

df = px.data.iris()
fig = px.scatter(
    df, x=df.sepal_length, y=df.sepal_width, color=df.species, size=df.petal_length
)

my_project_ui.put("my_plotly_fig", fig)

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
    range_x=[100, 100_000],
    range_y=[25, 90],
)

my_project_ui.put("my_anim_plotly_fig", my_anim_plotly_fig)

# %%
# PIL image:

# %%

my_pil_image = PIL.Image.new("RGB", (100, 100), color="red")
with io.BytesIO() as output:
    my_pil_image.save(output, format="png")

my_project_ui.put("my_pil_image", my_pil_image)

# %%
# Storing scikit-learn models and pipelines
# =========================================
#
# As skore is developed by `Probabl <https://probabl.ai>`_, the spin-off of
# scikit-learn, skore treats scikit-learn models and pipelines as first-class citizens.
#
# First of all, you can store a scikit-learn model:

# %%

my_model = Lasso(alpha=2)
my_project_ui.put("my_model", my_model)
my_model

# %%
# You can also store scikit-learn pipelines:

# %%

my_pipeline = Pipeline(
    [("standard_scaler", StandardScaler()), ("lasso", Lasso(alpha=2))]
)
my_project_ui.put("my_pipeline", my_pipeline)
my_pipeline

# %%
# Moreover, you can store fitted scikit-learn pipelines:

# %%

diabetes = load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
my_pipeline.fit(X, y)

my_project_ui.put("my_fitted_pipeline", my_pipeline)
my_pipeline

# %%
# Cleanup the project
# -------------------
#
# Remove the temporary directory:
temp_dir.cleanup()
