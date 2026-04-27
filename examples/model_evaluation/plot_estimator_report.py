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
# manufacturing companies paid a medical doctors or hospitals, in order to detect
# potential conflict of interest.

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
# task is quite imbalanced. It means that we have to be careful when selecting a set
# of statistical metrics to evaluate the classification performance of our predictive
# model. In addition, we see that the class labels are not specified by an integer
# 0 or 1 but instead by a string "allowed" or "disallowed".
#
# For our application, the label of interest is "allowed".
pos_label, neg_label = "allowed", "disallowed"

# %%
# Before training a predictive model, we need to split our dataset into a training
# and a validation set.
from skore import train_test_split

# If you have many dataframes to split on, you can always ask train_test_split to return
# a dictionary. Remember, it needs to be passed as a keyword argument!
split_data = train_test_split(X=df, y=y, random_state=42, as_dict=True)

# %%
# By the way, notice how skore's :func:`~skore.train_test_split` automatically warns us
# for a class imbalance.
#
# Now, we need to define a predictive model. Hopefully, `skrub` provides a convenient
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
# Getting insights from our estimator
# ===================================
#
# Introducing the :class:`skore.EstimatorReport` class
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now, we would be interested in getting some insights from our predictive model.
# One way is to use the :class:`skore.EstimatorReport` class. This constructor will
# detect that our estimator is unfitted and will fit it for us on the training data.
from skore import EstimatorReport

report = EstimatorReport(estimator, **split_data, pos_label=pos_label)
report

# %%
#
# Once the report is created, we get some information regarding the available tools
# allowing us to get some insights from our specific model on our specific task by
# calling the :meth:`~skore.EstimatorReport.help` method.
report.help()

# %%
#
# Be aware that we can access the help for each individual sub-accessor. For instance:
report.metrics.help()

# %%
#
# Metrics computation and repeated evaluation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# At this point, we might be interested to have a first look at the statistical
# performance of our model on the validation set that we provided. We can access it
# by calling any of the metrics displayed above. Since we are greedy, we want to get
# several metrics at once and we will use the
# :meth:`~skore.EstimatorReport.metrics.summarize` method.
import time

start = time.time()
metric_report = report.metrics.summarize().frame()
end = time.time()
metric_report

# %%
print(f"Time taken to compute the metrics: {end - start:.2f} seconds")

# %%
#
# On large enough data, getting predictions is often the expensive step. The report
# keeps intermediate results in memory for the same session, so when we ask for the
# same :meth:`~skore.EstimatorReport.metrics.summarize` again, it can complete much
# faster. Let's request the same summary a second time.

start = time.time()
metric_report = report.metrics.summarize().frame()
end = time.time()
metric_report

# %%
print(f"Time taken to compute the metrics: {end - start:.2f} seconds")

# %%
#
# Note that when the model is fitted or the predictions are computed,
# we additionally store the time the operation took:
report.metrics.timings()

# %%
#
# Since we obtain a pandas dataframe, we can also use the plotting interface of
# pandas.
ax = metric_report.plot.barh()
_ = ax.set_title("Metrics report")

# %%
#
# Another metric on the test set, such as log loss, can reuse the same underlying
# predictions if they were already required for a previous call.

start = time.time()
log_loss = report.metrics.log_loss()
end = time.time()
log_loss

# %%
print(f"Time taken to compute the log loss: {end - start:.2f} seconds")

# %%
#
# By default, the metrics are computed on the test set only. However, if a training set
# is provided, we can also compute the metrics by specifying the `data_source`
# parameter.
report.metrics.log_loss(data_source="train")

# %%
#
# Custom metrics also go through the same path: they receive `y_true` and `y_pred`
# as the first two arguments, and the report supplies predictions consistently with
# built-in metrics. The callable can take any other arguments. Let's see an example.


def operational_decision_cost(y_true, y_pred, amount):
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
# In our use case, we have a operational decision to make that translate the
# classification outcome into a cost. It translate the confusion matrix into a cost
# matrix based on some amount linked to each sample in the dataset that are provided to
# us. Here, we randomly generate some amount as an illustration.
import numpy as np

rng = np.random.default_rng(42)
amount = rng.integers(low=100, high=1000, size=len(split_data["y_test"]))

report.metrics.add(
    metric=operational_decision_cost,
    response_method="predict",
    amount=amount,
)

cost = report.metrics.summarize(metric="operational_decision_cost")
cost.frame()

# %%
#
# By the way, skore caches the model predictions. It is really handy because it means
# that we can compute some additional metrics without having to recompute the
# the predictions.
report.metrics.summarize(
    metric=["precision", "recall", "operational_decision_cost"],
).frame()

# %%
#
# Effortless one-liner plotting
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :class:`skore.EstimatorReport` class also provides a plotting interface that
# allows to plot *defacto* the most common plots. As for the metrics, we only
# provide the meaningful set of plots for the provided estimator.
report.metrics.help()

# %%
#
# Let's start by plotting the ROC curve for our binary classification task.
display = report.metrics.roc()
display.plot()

# %%
#
# The plot functionality is built upon the scikit-learn display objects. We return
# those display (slightly modified to improve the UI) in case we want to tweak some
# of the plot properties. We can have quick look at the available attributes and
# methods by calling the ``help`` method or simply by printing the display.
display.help()

# %%
fig = display.plot()
fig.axes[0].set_title("Example of a ROC curve")
fig

# %%
#
# Similarly to the metrics, repeated calls for the same ROC display can be much
# faster in the same session once the underlying values have been computed.
start = time.time()
# we already trigger the computation of the predictions in a previous call
display = report.metrics.roc()
fig = display.plot()
end = time.time()
fig

# %%
print(f"Time taken to compute the ROC curve: {end - start:.2f} seconds")

# %%
# Visualizing the confusion matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Another useful visualization for classification tasks is the confusion matrix,
# which shows the counts of correct and incorrect predictions for each class.

# %%
# Let's first start with a basic confusion matrix:
cm_display = report.metrics.confusion_matrix()
cm_display.plot()

# %%
# In binary classification, a confusion matrix depends on the decision threshold used
# to convert predicted probabilities into class labels. By default, skore uses a
# threshold of 0.5, but confusion matrices are actually computed at every threshold
# internally.

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
