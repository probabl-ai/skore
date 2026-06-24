"""
.. _example_cache_mechanism:

===============
Cache mechanism
===============

This example shows how :class:`~skore.EstimatorReport` and
:class:`~skore.CrossValidationReport` use caching to speed up computations.
"""

# %%
# Generating some data
# ====================
#
# In this toy example, we create a large synthetic classification dataset that will let
# us see speed improvements easily.
import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=150_000, return_X_y=True)
X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])

# %%
# Here is what the training data looks like:
from skrub import TableReport

TableReport(X)

# %%
# And the target training data:
TableReport(y)

# %%
# We build a model using :func:`skrub.tabular_pipeline`: it is a simple predictive
# model that also performs basic feature engineering.
from skrub import tabular_pipeline

model = tabular_pipeline("classifier")
model

# %%
# Caching the predictions for fast metric computation
# ===================================================
#
# Let's explore how :class:`~skore.EstimatorReport` uses caching to speed up
# predictions.
from skore import evaluate

report = evaluate(model, X, y, pos_label=1)
report.help()

# %%
#
# We compute the accuracy on our test set and measure how long it takes:
import time

start = time.time()
report.metrics.accuracy()
end = time.time()
print(f"Time taken: {end - start:.3f} seconds")

# %%
#
# For comparison, here's how scikit-learn computes the same accuracy score:
from sklearn.metrics import accuracy_score

start = time.time()
accuracy_score(report.y_test, report.estimator_.predict(report.X_test))
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# %%
# skore outputs the result much faster than scikit-learn. How can this be?
# The answer lies in the EstimatorReport's state. When the EstimatorReport is created,
# it computes the model predictions, and caches them:
report._cache

# %%
# The cache stores predictions by type and data source. This means that computing
# metrics that use the same type of predictions will be faster.

# %%
# Caching plots
# =============
#
# The cache also speeds up plot generation. For instance, this is how long it takes
# to plot the ROC curve:
start = time.time()
report.metrics.roc().plot()
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# %%
# Now do it a second time:
start = time.time()
report.metrics.roc().plot()
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# %%
# Caching with :class:`~skore.CrossValidationReport`
# ==================================================
#
# Here we will demonstrate that :class:`~skore.CrossValidationReport` also benefits
# from caching.
report = evaluate(model, X=X, y=y, splitter=3, n_jobs=3)
report.help()

# %%
#
# A :class:`~skore.CrossValidationReport` is essentially a list of
# :class:`~skore.EstimatorReport`, one for each split, so caching on the splits
# makes the calculation on the :class:`~skore.CrossValidationReport` faster as well.
start = time.time()
report.metrics.summarize().frame()
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# The subsequent calls are even faster because the metrics themselves are cached:
start = time.time()
report.metrics.summarize().frame()
end = time.time()
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# By keeping the estimator together with the data, we are able to trade off some
# memory space for faster operations.
