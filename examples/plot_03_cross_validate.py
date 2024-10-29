"""
.. _example_cross_validate:

==========================
Enhancing cross-validation
==========================

This example illustrates the use of :func:`~skore.cross_validate` to get
assistance when developing your ML/DS projects.
"""

# %%
import subprocess

# remove the skore project if it already exists
subprocess.run("rm -rf my_project_cv.skore".split())

# create the skore project
subprocess.run("python3 -m skore create my_project_cv".split())


# %%
from skore import load

my_project_gs = load("my_project_cv.skore")

# %%
from sklearn import datasets, linear_model
from skore.cross_validate import cross_validate

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()

cv_results = cross_validate(lasso, X, y, cv=3, project=my_project_gs)

my_project_gs.get_item("cross_validation").plot

# %%
# Because Plotly graphs currently do not yet properly render in our docs engine due to `a bug in Plotly <https://github.com/plotly/plotly.py/issues/4828>`_,
# we also show a screenshot below.
# Alternatively, if you zoom in or out in your browser window, the Plotly graph should display properly again.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("plot_03_cross_validate_plot_screenshot.png")
fig, ax = plt.subplots()
# fig.tight_layout(pad=0.01)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
ax.axis("off")
ax.imshow(img)
