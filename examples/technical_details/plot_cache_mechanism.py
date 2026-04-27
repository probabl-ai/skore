"""
.. _example_cache_mechanism:

====================================
Fast repeated metrics and evaluation
====================================

This example shows that :class:`~skore.EstimatorReport` and
:class:`~skore.CrossValidationReport` avoid redundant work when you compute metrics
or displays several times, so the second call is often much faster than the first.
"""

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
# :class:`~skore.EstimatorReport` and repeated evaluation
# =======================================================
#
# We use `skrub` to create a simple predictive model that handles our dataset's
# challenges.
from skrub import tabular_pipeline

model = tabular_pipeline("classifier")
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
# First and second calls to a metric
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We build an :class:`~skore.EstimatorReport` and time how long successive metric
# calls take.
from skore import EstimatorReport

report = EstimatorReport(
    model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    pos_label="allowed",
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
# Now, we compute the accuracy again through the same report:
start = time.time()
result = report.metrics.accuracy()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# The second calculation is much faster, because the report does not repeat the
# expensive ``predict`` work when the same information is still available for this
# session.

# %%
# A different metric that needs the same predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's try the precision metric:
start = time.time()
result = report.metrics.precision()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# It typically stays fast, because the same type of test-set predictions is reused
# where possible.

# %%
# Another data source
# ^^^^^^^^^^^^^^^^^^^
#
# The first time we ask for a training-set metric, the model must be run on the
# training set as well. Later calls on that data source also benefit from reuse.
start = time.time()
result = report.metrics.log_loss(data_source="train")
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
# Plots
# ^^^^^
#
# Displays (for example a ROC curve) also benefit: the first request builds the
# underlying arrays; a second request for the same display is quick.

start = time.time()
display = report.metrics.roc()
display.plot()
end = time.time()

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
start = time.time()
display = report.metrics.roc()
display.plot()
end = time.time()

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# We can still customize the display (for example style) and plot again; the
# evaluation work behind the same metric does not need to be redone in full.
display.set_style(relplot_kwargs={"color": "tab:orange"})
_ = display.plot()

# %%
# Cross-validation: :class:`~skore.CrossValidationReport`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A :class:`~skore.CrossValidationReport` uses one
# :class:`~skore.EstimatorReport` per split, so the same idea applies: the first
# heavy summary of metrics walks every fold; a second run reuses work where possible.
from skore import CrossValidationReport

report = CrossValidationReport(model, X=df, y=y, splitter=5, n_jobs=4)
report.help()

# %%
#
# The first call to a full summary of metrics can take a while because each fold
# is evaluated.
start = time.time()
result = report.metrics.summarize().frame()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# The second call is typically much faster.
start = time.time()
result = report.metrics.summarize().frame()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")
