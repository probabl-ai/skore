"""
========================
Basic usage of ``skore``
========================

*This is work in progress.*

This example builds on top of the :ref:`getting_started` guide.
"""

# %%
from skore import load

# project = load("project.skore")

# %%
# project.put("my_int", 3)

# %%
import sklearn
import matplotlib.pyplot as plt

# %%
import numpy as np
import pandas as pd

my_df = pd.DataFrame(np.random.randn(3, 3))
# project.put("my_df", my_df)

# %%
sklearn.__version__
plt.plot([0, 1, 2, 3], [0, 1, 2, 3])
plt.show()

# %%
import subprocess

subprocess.run("rm -rf my_project_doc.skore".split())
subprocess.run("python3 -m skore create my_project_doc".split())

# %%
from skore import load
from sklearn import datasets, linear_model
from skore.cross_validate import cross_validate

my_project_doc = load("my_project_doc")

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()

cv_results = cross_validate(lasso, X, y, cv=3, project=my_project_doc)

my_project_doc.get_item("cross_validation").plot
