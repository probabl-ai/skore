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

my_project = skore.create("quick_start", overwrite=True)

# %%
# This will create a skore project directory named ``quick_start.skore`` in your
# current working directory.

# %%
# Evaluate your model using skore's :class:`~skore.CrossValidationReporter`:

# %%
from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
clf = SVC(kernel="linear", C=1, random_state=0)

reporter = skore.CrossValidationReporter(clf, X, y, cv=5)

# %%
# Store the results in the skore project:

# %%
my_project.put("cv_reporter", reporter)

# %%
# Display results in your notebook:

# %%
reporter.plots.scores

# %%
reporter.plots.timing_normalized

# %%
# Finally, from your shell (in the same directory), start the UI:
#
# .. code-block:: bash
#
#   $ skore launch "quick_start"
#
# This will open skore-ui in a browser window.
#
# .. image:: https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/2024_12_12_skore_demo_comp.gif
#   :alt: Getting started with ``skore`` demo
#
# .. admonition:: What's next?
#
#    For a more in-depth guide, see our :ref:`example_skore_product_tour` page!
