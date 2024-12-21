"""
============================================
Get insights from any scikit-learn estimator
============================================

This example shows how the :class:`skore.EstimatorReport` class can be used to
quickly get insights from any scikit-learn estimator.
"""

# %%
#
# Let's take a non-trivial dataset containing some categorical, date, and numerical
# features: the employee salary dataset. This dataset is made available from the
# `skrub` library.
import pandas as pd
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
df = dataset.X
y = dataset.y
# convert to get proper datetime dtype
df["date_first_hired"] = pd.to_datetime(df["date_first_hired"])
df

# %%
y

# %%
#
# The aim with this dataset is to predict the annual salary of an employee based on
# their personal and job information.
#
# Let's first split the dataset into a training and validation set.
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df, y, random_state=42)

# %%
#
# We can use `skrub` and in particular the :class:`skrub.tabular_learner` function
# to quickly to get a first model that will be a good starting point.
from skrub import tabular_learner

estimator = tabular_learner("regressor").fit(X_train, y_train)
estimator

# %%
#
# We see that the model that we get is able to handle all the different types of data.
# Now, let's use the :class:`skore.EstimatorReport` class to get insights from this
# model. Note that we call the `from_fitted_estimator` method, since we already fitted
# our estimator. We will see later that `skore` also provides a
# `from_unfitted_estimator` method, which will first fit the model on some training
# data and then provide similar insights.
from skore import EstimatorReport

reporter = EstimatorReport.from_fitted_estimator(estimator, X=X_val, y=y_val)
reporter

# %%
#
# Now that our report is created, you can call the `help` method to get more information
# on the different available methods to get insights from the reporter.
reporter.help()
