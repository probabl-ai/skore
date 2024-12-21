"""
============================================
Get insights from any scikit-learn estimator
============================================

This example shows how the :class:`skore.EstimatorReport` class can be used to
quickly get insights from any scikit-learn estimator.
"""

# %%
from skrub.datasets import fetch_open_payments

dataset = fetch_open_payments()
df = dataset.X
y = dataset.y
df

# %%
y

# %%
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df, y, random_state=42)

# %%
from skrub import tabular_learner

estimator = tabular_learner("classifier").fit(X_train, y_train)
estimator

# %%
from skore import EstimatorReport

reporter = EstimatorReport.from_fitted_estimator(estimator, X=X_val, y=y_val)
reporter

# %%
reporter.plot.roc(positive_class="allowed")

# %%
