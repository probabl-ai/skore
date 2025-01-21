"""
.. _example_cross_validate:

================
Cross-validation
================

This example illustrates the motivation and the use of skore's
:class:`skore.CrossValidationReporter` to get assistance when developing ML/DS projects.

.. warning ::

    **Deprecation Notice**:
    :class:`skore.CrossValidationReporter` is deprecated in favor of :class:`skore.CrossValidationReport`.
"""

# %%
# Creating and loading the skore project
# ======================================

# %%
# We start by creating a temporary directory to store our project so that we can
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

my_project = skore.open(temp_dir_path / "my_project")

# %%
# Cross-validation in scikit-learn
# ================================
#
# Scikit-learn holds two functions for cross-validation:
#
# * :func:`sklearn.model_selection.cross_val_score`
# * :func:`sklearn.model_selection.cross_validate`
#
# Essentially, :func:`sklearn.model_selection.cross_val_score` runs cross-validation for
# single metric evaluation, while :func:`sklearn.model_selection.cross_validate` runs
# cross-validation with multiple metrics and can also return extra information such as
# train scores, fit times, and score times.
#
# Hence, in skore, we are more interested in the
# :func:`sklearn.model_selection.cross_validate` function as it allows to do
# more than the historical :func:`sklearn.model_selection.cross_val_score`.
#
# Let us illustrate cross-validation on a multi-class classification task.

# %%
from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
clf = SVC(kernel="linear", C=1, random_state=0)

# %%
# Single metric evaluation using :func:`sklearn.model_selection.cross_validate`:

# %%
from sklearn.model_selection import cross_validate as sklearn_cross_validate

cv_results = sklearn_cross_validate(clf, X, y, cv=5)
print(f"test_score: {cv_results['test_score']}")

# %%
# Multiple metric evaluation using :func:`sklearn.model_selection.cross_validate`:

# %%
import pandas as pd

cv_results = sklearn_cross_validate(
    clf,
    X,
    y,
    cv=5,
    scoring=["accuracy", "precision_macro"],
)
test_scores = pd.DataFrame(cv_results)[["test_accuracy", "test_precision_macro"]]
test_scores

# %%
# In scikit-learn, why do we recommend using ``cross_validate`` over ``cross_val_score``?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Here, for the :class:`~sklearn.svm.SVC`, the default score is the accuracy.
# If the users want other scores to better understand their model such as the
# precision and the recall, they can specify it which is very convenient.
# Otherwise, they would have to run several
# :func:`sklearn.model_selection.cross_val_score` with different ``scoring``
# parameters each time, which leads to more unnecessary compute.
#
# Why do we recommend using skore's ``CrossValidationReporter`` over scikit-learn's ``cross_validate``?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the example above, what if the users ran scikit-learn's
# :func:`sklearn.model_selection.cross_validate` but forgot to manually add a
# crucial score for their use case such as the recall?
# They would have to re-run the whole cross-validation experiment by adding this
# crucial score, which leads to more compute.

# %%
# Cross-validation in skore
# =========================
#
# In order to assist its users when programming, skore has implemented a
# :class:`skore.CrossValidationReporter` class that wraps scikit-learn's
# :func:`sklearn.model_selection.cross_validate`, to provide more
# context and facilitate the analysis.
#
# Classification task
# ^^^^^^^^^^^^^^^^^^^
#
# Let us continue with the same use case.

# %%
reporter = skore.CrossValidationReporter(clf, X, y, cv=5)
reporter.plots.scores

# %%
# Skore's :class:`~skore.CrossValidationReporter` advantages are the following:
#
# * By default, it computes several useful scores without the need to
#   manually specify them. For classification, one can observe that it computed the
#   accuracy, the precision, and the recall.
#
# * We automatically get some interactive Plotly graphs to better understand how our
#   model behaves depending on the split. For example:
#
#   * We can compare the fitting and scoring times together for each split.
#
#   * Moreover, we can focus on the times per data points as the train and
#     test splits usually have a different number of samples.
#
#   * We can compare the accuracy, precision, and recall scores together for each
#     split.

# %%
# Regression task
# ^^^^^^^^^^^^^^^

# %%
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso

X, y = load_diabetes(return_X_y=True)
lasso = Lasso()

reporter = skore.CrossValidationReporter(lasso, X, y, cv=5)
reporter.plots.scores

# %%
# We can put the reporter in the project, and retrieve it as is:
my_project.put("cross_validation_reporter", reporter)

reporter = my_project.get("cross_validation_reporter")
reporter.plots.scores

# %%
# Cleanup the project
# -------------------
#
# Removing the temporary directory:

# %%
temp_dir.cleanup()
