"""
.. _example_quick_start:

===========
Quick start
===========
"""

# %%
# From your Python code, create and load a skore :class:`~skore.Project`:

# %%
import skore

my_project = skore.open("quick_start", overwrite=True)

# %%
# This will create a skore project directory named ``quick_start.skore`` in your
# current working directory and overwrite any pre-existing project with the
# same path (which you might not want to do that depending on your use case).

# %%
# Evaluate your model using skore's :class:`~skore.CrossValidationReporter`:

# %%
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
clf_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])

reporter = skore.CrossValidationReporter(clf_pipeline, X, y, cv=5)

# %%
# Store the results in the skore project:

# %%
my_project.put("cv_reporter", reporter)

# %%
# Display some results in your notebook:

# %%
reporter.plots.timing

# %%
# .. admonition:: What's next?
#
#    For a more in-depth guide, see our :ref:`example_skore_product_tour` page!
