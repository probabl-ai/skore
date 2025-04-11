"""
.. _example_cache_mechanism:

===============
Cache mechanism
===============

This example shows how :class:`~skore.EstimatorReport` and
:class:`~skore.CrossValidationReport` use caching to speed up computations.
"""

# %%
#
# We set some environment variables to avoid some spurious warnings related to
# parallelism.
import os

os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"

# %%
# Loading some data
# =================
#
# First, we load a dataset from `skrub`. Our goal is to predict if a company paid a
# physician. The ultimate goal is to detect potential conflict of interest when it comes
# to the actual problem that we want to solve.
from skrub.datasets import fetch_open_payments

dataset = fetch_open_payments()
df = dataset.X
y = dataset.y

# %%
from skrub import TableReport

TableReport(df)

# %%
import pandas as pd

TableReport(pd.DataFrame(y))

# %%
#
# The dataset has over 70,000 records with only categorical features.
# Some categories are not well defined.

# %%
# Caching with :class:`~skore.EstimatorReport` and :class:`~skore.CrossValidationReport`
# ======================================================================================
#
# We use `skrub` to create a simple predictive model that handles our dataset's
# challenges.
from skrub import tabular_learner

model = tabular_learner("classifier")
model


# %%
#
# This model handles all types of data: numbers, categories, dates, and missing values.
# Let's train it on part of our dataset.
from skore import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)
# Let's keep a completely separate dataset
X_train, X_external, y_train, y_external = train_test_split(
    X_train, y_train, random_state=42
)

# %%
# Caching the predictions for fast metric computation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# First, we focus on :class:`~skore.EstimatorReport`, as the same philosophy will
# apply to :class:`~skore.CrossValidationReport`.
#
# Let's explore how :class:`~skore.EstimatorReport` uses caching to speed up
# predictions. We start by training the model:
from skore import EstimatorReport

report = EstimatorReport(
    model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)
report.help()

# %%
#
# We compute the accuracy on our test set and measure how long it takes:
import time

start = time.time()
result = report.metrics.accuracy()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# For comparison, here's how scikit-learn computes the same accuracy score:
from sklearn.metrics import accuracy_score

start = time.time()
result = accuracy_score(report.y_test, report.estimator_.predict(report.X_test))
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# Both approaches take similar time.
#
# Now, watch what happens when we compute the accuracy again with our skore estimator
# report:
start = time.time()
result = report.metrics.accuracy()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# The second calculation is instant! This happens because the report saves previous
# calculations in its cache. Let's look inside the cache:
report._cache

# %%
# The cache stores predictions by type and data source. This means that computing
# metrics that use the same type of predictions will be faster.
# Let's try the precision metric:
start = time.time()
result = report.metrics.precision()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
# We observe that it takes only a few milliseconds to compute the precision because we
# don't need to re-compute the predictions and only have to compute the precision
# metric itself.
# Since the predictions are the bottleneck in terms of computation time, we observe
# an interesting speedup.

# %%
# Caching all the possible predictions at once
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can pre-compute all predictions at once using parallel processing:
report.cache_predictions(n_jobs=4)

# %%
#
# Now, all possible predictions are stored. Any metric calculation will be much faster,
# even on different data (like the training set):
start = time.time()
result = report.metrics.log_loss(data_source="train")
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
# Caching external data
# ^^^^^^^^^^^^^^^^^^^^^
#
# The report can also work with external data. We use `data_source="X_y"` to indicate
# that we want to pass those external data.
start = time.time()
result = report.metrics.log_loss(data_source="X_y", X=X_external, y=y_external)
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# The first calculation of the above cell is slower than when using the internal train
# or test sets because it needs to compute a hash of the new data for later retrieval.
# Let's calculate it again:
start = time.time()
result = report.metrics.log_loss(data_source="X_y", X=X_external, y=y_external)
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# It is much faster for the second time as the predictions are cached!
# The remaining time corresponds to the hash computation.
# Let's compute the ROC AUC on the same data:
start = time.time()
result = report.metrics.roc_auc(data_source="X_y", X=X_external, y=y_external)
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
# We observe that the computation is already efficient because it boils down to two
# computations: the hash of the data and the ROC-AUC metric.
# We save a lot of time because we don't need to re-compute the predictions.

# %%
# Caching for plotting
# ^^^^^^^^^^^^^^^^^^^^
#
# The cache also speeds up plots. Let's create a ROC curve:

import matplotlib.pyplot as plt

start = time.time()
display = report.metrics.roc(pos_label="allowed")
display.plot()
end = time.time()

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# The second plot is instant because it uses cached data:
start = time.time()
display = report.metrics.roc(pos_label="allowed")
display.plot()
end = time.time()

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# We only use the cache to retrieve the `display` object and not directly the matplotlib
# figure. It means that we can still customize the cached plot before displaying it:
display.plot(roc_curve_kwargs={"color": "tab:orange"})

# %%
#
# Be aware that we can clear the cache if we want to:
report.clear_cache()
report._cache

# %%
#
# It means that nothing is stored anymore in the cache.
#
# Caching with :class:`~skore.CrossValidationReport`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# :class:`~skore.CrossValidationReport` uses the same caching system for each fold
# in cross-validation by leveraging the previous :class:`~skore.EstimatorReport`:
from skore import CrossValidationReport

report = CrossValidationReport(model, X=df, y=y, cv_splitter=5, n_jobs=4)
report.help()

# %%
#
# Since a :class:`~skore.CrossValidationReport` uses many
# :class:`~skore.EstimatorReport`, we will observe the same behaviour as we previously
# exposed.
# The first call will be slow because it computes the predictions for each fold.
start = time.time()
result = report.metrics.report_metrics()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# But the subsequent calls are fast because the predictions are cached.
start = time.time()
result = report.metrics.report_metrics()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# Hence, we observe the same type of behaviour as we previously exposed.
