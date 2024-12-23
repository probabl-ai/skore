"""
.. _example_quick_start:

===========
Quick start
===========
"""

# %%
import skore

my_project = skore.create("quick_start", overwrite=True)

# %%
from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)
clf = SVC(kernel="linear", C=1, random_state=0)

reporter = skore.CrossValidationReporter(clf, X, y, cv=5)

# Store the results in the project
my_project.put("cv_reporter", reporter)

# Display the result in your notebook
reporter.plots.scores

# %%
# .. code-block:: bash
#
#   $ skore launch "my_project"
