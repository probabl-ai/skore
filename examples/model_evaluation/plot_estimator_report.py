"""
.. _example_estimator_report:

===============================================================
`EstimatorReport`: Get insights from any scikit-learn estimator
===============================================================

This example shows how the :class:`skore.EstimatorReport` class can be used to
quickly get insights from any scikit-learn estimator.
"""

# %%
# Loading our dataset and defining our estimator
# ==============================================
#
# First, we load a dataset from skrub. Our goal is to predict if a healthcare
# manufacturing company paid medical doctors or hospitals, in order to detect
# potential conflicts of interest.

# %%
from skrub.datasets import fetch_open_payments

dataset = fetch_open_payments()
df = dataset.X
y = dataset.y

# %%
from skrub import TableReport

TableReport(df)

# %%
TableReport(y.to_frame())

# %%
# Looking at the distributions of the target, we observe that this classification
# task is quite imbalanced. This means that we have to be careful when selecting a set
# of statistical metrics to evaluate the classification performance of our predictive
# model. In addition, we see that the class labels are not specified by an integer
# 0 or 1 but instead by a string "allowed" or "disallowed".
#
# For our application, the label of interest is "allowed".
pos_label, neg_label = "allowed", "disallowed"

# %%
# Let's create a predictive model. Thankfully, `skrub` provides a convenient
# function (:func:`skrub.tabular_pipeline`) when it comes to getting strong baseline
# predictive models with a single line of code. As its feature engineering is generic,
# it does not provide some handcrafted and tailored feature engineering but still
# provides a good starting point.
#
# So let's create a classifier for our task.
from skrub import tabular_pipeline

estimator = tabular_pipeline("classifier")
estimator

# %%
# Introducing the :class:`EstimatorReport`
# ========================================
#
# Let's gather some insights from our predictive model.
# We can use :func:`skore.evaluate` for this: the function will perform a train-test
# split and create a :class:`~skore.EstimatorReport` containing the model fitted on
# the training data, ready to investigate.

from skore import evaluate

# Reserve 20% of the data for the test set
report = evaluate(estimator, X=df, y=y, pos_label=pos_label, splitter=0.2)
report

# %%
#
# Once the report is created, we get some information regarding the available tools
# allowing us to get some insights on our model by calling the
# :meth:`~skore.EstimatorReport.help` method.
report.help()

# %%
#
# Be aware that we can access the help for each individual sub-accessor. For instance:
report.metrics.help()

# %%
#
# Measuring model performance
# ===========================
#
# Let's have a first look at the statistical performance of our model. skore knows
# that we are doing classification, and can give us an array of classic ML metrics,
# all at once, with :meth:`~skore.EstimatorReport.metrics.summarize`:
import time

start = time.time()
metric_report = report.metrics.summarize().frame()
end = time.time()
metric_report

# %%
print(f"Time taken to compute the metrics: {end - start:.2f} seconds")

# %%
#
# Since the output is a pandas dataframe, we can also use the plotting interface of
# pandas.
ax = metric_report.plot.barh()
_ = ax.set_title("Metrics report")

# %%
#
# An interesting feature of the :class:`skore.EstimatorReport` is its caching mechanism.
# Indeed, when we have a large enough dataset, computing the predictions for a model can
# be expensive. To amortize this cost, the report will cache the predictions when
# it is first created; this way, calculations that need the model predictions can get
# them from the cache and save a lot of time. This is why the metrics computation above
# is so fast.

# %%
#
# When the model is fitted or the predictions are computed,
# we additionally store the time the operation took:
report.metrics.timings()

# %%
#
# By default, the metrics are computed on the test set only, but we can also compute
# them on the train set:
report.metrics.log_loss(data_source="train")


# %%
#
# Defining custom metrics
# =======================
#
# skore can compute user-defined metrics as well. It accepts metrics in the form of
# scikit-learn scorers, i.e. functions taking `estimator`, `X` and `y` (and optional
# keyword arguments). Let's take a look at an example.


def operational_decision_cost(y_true, y_pred, *, amount):
    mask_true_positive = (y_true == pos_label) & (y_pred == pos_label)
    mask_true_negative = (y_true == neg_label) & (y_pred == neg_label)
    mask_false_positive = (y_true == neg_label) & (y_pred == pos_label)
    mask_false_negative = (y_true == pos_label) & (y_pred == neg_label)
    fraudulent_refuse = mask_true_positive.sum() * 50
    fraudulent_accept = -amount[mask_false_negative].sum()
    legitimate_refuse = mask_false_positive.sum() * -5
    legitimate_accept = (amount[mask_true_negative] * 0.02).sum()
    return fraudulent_refuse + fraudulent_accept + legitimate_refuse + legitimate_accept


# %%
#
# In our example use case, each classification decision we make has a cost.
# The function above models this by translating the confusion matrix into a cost
# matrix; this cost also depends on an extra parameter named ``amount``, to illustrate
# that skore can handle custom metrics with non-standard arguments.
# Let's test adding this metric to our report.
import numpy as np
from sklearn.metrics import make_scorer

rng = np.random.default_rng(42)
amount = rng.integers(low=100, high=1000, size=len(report.y_test))

# We use `make_scorer` to convert the metric to the right format (a function
# that takes `estimator`, `X`, `y`)
report.metrics.add(metric=make_scorer(operational_decision_cost, amount=amount))

# %%
#
# Our custom metric is now registered in the report, and will be shown in the summary.
# In fact, since the underlying metric function takes `y_pred` as input, skore can use
# the cached predictions again to speed up the computation.

# The metric name is derived from the function name unless it is explicitly given
report.metrics.summarize().frame()

# %%
#
# Effortless one-liner plotting
# =============================
#
# The :class:`skore.EstimatorReport` class also implements a number of the most common
# data science plots.
# As for the metrics, we only provide the meaningful set of plots for the provided
# estimator.
report.metrics.help()

# %%
#
# Let's plot the ROC curve for our binary classification task.
display = report.metrics.roc()
display.plot()

# %%
#
# The plot functionality is built upon the scikit-learn Display objects. We return
# those Display objects (slightly modified to improve the UI) in case we want to tweak some
# of the plot properties. We can have a quick look at the available attributes and
# methods by calling the ``help`` method.
display.help()

# %%
fig = display.plot()
fig.axes[0].set_title("Example of a ROC curve")
fig

# %%
#
# Similarly to the metrics, the cache allows us to avoid recomputing the model
# predictions, which speeds up the display generation.
start = time.time()
display = report.metrics.roc()
_ = display.plot()
end = time.time()
print(f"Time taken to compute the ROC curve: {end - start:.2f} seconds")

# %%
# You can learn more about the cache system in the corresponding example:
# :ref:`example_cache_mechanism`.

# %%
# Visualizing the confusion matrix
# ================================
#
# Another useful visualization for classification tasks is the confusion matrix,
# which shows the counts of correct and incorrect predictions for each class.

# %%
# Let's start with a basic confusion matrix:
cm_display = report.metrics.confusion_matrix()
cm_display.plot()

# %%
# In binary classification, a confusion matrix depends on the decision threshold used
# to convert predicted probabilities into class labels. By default, skore uses a
# threshold of 0.5, but confusion matrices are actually computed at every threshold
# internally.
#
# To visualize the confusion matrix at a different threshold, use the
# ``threshold_value`` parameter. For example, a threshold of 0.3 will classify
# more samples as positive:
cm_display.plot(threshold_value=0.3)

# %%
# We can normalize the confusion matrix to get percentages instead of raw counts.
# Here we normalize by true labels (rows):
cm_display.plot(normalize="true")

# %%
# More plotting options are available via ``heatmap_kwargs``, which are passed to
# seaborn's heatmap. For example, we can customize the colormap and number format:
cm_display.set_style(heatmap_kwargs={"cmap": "Greens", "fmt": ".2e"})
cm_display.plot()

# %%
# Finally, the confusion matrix can also be exported as a pandas DataFrame for further
# analysis:
cm_display.frame()

# %%
# .. seealso::
#
#   For using the :class:`~skore.EstimatorReport` to inspect your models,
#   see :ref:`example_feature_importance`.
