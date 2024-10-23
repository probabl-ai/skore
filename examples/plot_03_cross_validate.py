"""
===================================
3) Using ``skore``'s cross validate
===================================

This example illustrates the use of :func:`~skore.cross_validate`.
"""

# %%
import subprocess

# remove the project if it already exists
subprocess.run("rm -rf my_project_cv.skore".split())

# create the project
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
