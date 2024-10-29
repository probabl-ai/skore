"""
.. _example_cross_validate:

==========================
Enhancing cross-validation
==========================

This example illustrates the motivation and the use of
:func:`~skore.cross_validate` to get assistance when developing your
ML/DS projects.
"""


# %%
import subprocess

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.model_selection import cross_validate as sklearn_cross_validate

from skore import load
import skore.cross_validate


# %%
# Creating and loading the skore project
# ======================================

# %%

# remove the skore project if it already exists
subprocess.run("rm -rf my_project_cv.skore".split())

# create the skore project
subprocess.run("python3 -m skore create my_project_cv".split())


# %%
my_project_gs = load("my_project_cv.skore")

# %%
# Cross-validation in scikit-learn
# ================================
#
# Scikit-learn holds two functions for cross-validation:
#
# * ``cross_val_score``: `link <https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.cross_val_score.html>`_
# * ``cross_validate``: `link <https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.cross_validate.html>`_
#
# Essentially, ``cross_val_score`` runs cross-validation for single metric
# evaluation, while ``cross_validate`` runs cross-validation with multiple
# metrics and can also return extra information such as train scores, fit times, and score times.
#
# Hence, in skore, we are more interested in the ``cross_validate`` function as
# it allows to do more than the historical ``cross_val_score``.
#
# Let us illustrate cross-validation on a multi-class classification task.

# %%
X, y = datasets.load_iris(return_X_y=True)
clf = svm.SVC(kernel='linear', C=1, random_state=0)

# %%
# Single metric evaluation using ``cross_validate``:

# %%
cv_results = sklearn_cross_validate(clf, X, y, cv=5)
cv_results['test_score']

# %%
# Multiple metric evaluation using ``cross_validate``:

# %%
scores = sklearn_cross_validate(
    clf, X, y, cv=5,
    scoring=['accuracy', 'precision_macro'],
)
print(scores['test_accuracy'])
print(scores['test_precision_macro'])

# %%
# In scikit-learn, why do we recommend using ``cross_validate`` over ``cross_val_score``?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Here, for the
# `SVC <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_,
# the default score is the accuracy.
# If the users want other scores to better understand their model such as the
# precision and the recall, they can specify it which is very convenient.
# Otherwise, they would have to run several ``cross_val_score`` with different
# ``scoring`` parameters each time, which leads to more unnecessary compute.
#
# Why do we recommend using skore's ``cross_validate`` over scikit-learn's?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the example above, what if the users ran scikit-learn's ``cross_validate``
# but forgot to manually add a crucial score for their use case such as the
# recall?
# They would have to re-run all the cross-validation experiments by adding this
# crucial score, which leads to more compute.

# %%
# Cross-validation in skore
# =========================
#
# Classification task
# ^^^^^^^^^^^^^^^^^^^
#
# Let us continue with the same use case.

# %%
cv_results = skore.cross_validate(clf, X, y, cv=5, project=my_project_gs)

my_project_gs.get_item("cross_validation").plot

# %%
# Skore's ``cross_validate`` advantages are the following:
#
# * By default, it computes several useful scores without the need for the user to specify it. For classification, you can observe that it computed the accuracy, the precision, and the recall.
#
# * You automatically get some interactive plots to better understand how your model behaves depending on the split. For example:
#
#   * You can compare the fitting and testing times together for each split.
#
#   * You can compare the accuracy, precision, and recall scores together for each split.
#
# * The results and plots are automatically saved in your skore project, so that you can visualize them later in the UI for example.

# %%
# Regression task
# ^^^^^^^^^^^^^^^

# %%
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()

cv_results = skore.cross_validate(lasso, X, y, cv=5, project=my_project_gs)

my_project_gs.get_item("cross_validation").plot

# %%
# A short note on Plotly and Sphinx auto-examples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Because Plotly graphs currently do not properly render in our docs engine
# due to `a bug in Plotly <https://github.com/plotly/plotly.py/issues/4828>`_,
# we also display a screenshot below (using 3 splits).
# Alternatively, if you zoom in or out in your browser window, the Plotly graph
# should display properly.

# %%
img = mpimg.imread("plot_03_cross_validate_plot_screenshot.png")
fig, ax = plt.subplots(layout="constrained")
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
ax.axis("off")
ax.imshow(img)
plt.show()
