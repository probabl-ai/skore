"""
============================================
Get insights from any scikit-learn estimator
============================================

This example shows how the :class:`skore.EstimatorReport` class can be used to
quickly get insights from any scikit-learn estimator.
"""

# %%
#
# TODO: we need to describe the aim of this classification problem.
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
pos_label, negative_class = "allowed", "disallowed"

# %%
# Before to train a predictive model, we need to split our dataset into a training
# and a validation set.
from skore import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)

# %%
# TODO: we have a perfect case to show useful feature of the `train_test_split`
# function from `skore`.
#
# Now, we need to define a predictive model. Hopefully, `skrub` provides a convenient
# function (:func:`skore.tabular_learner`) when it comes to get strong baseline
# predictive models with a single line of code. Of course, it does not handcraft some
# specific feature engineering but it provides a good starting point.
#
# So let's create a classifier for our task and fit it on the training set.
from skrub import tabular_learner

estimator = tabular_learner("classifier").fit(X_train, y_train)
estimator

# %%
#
# Introducing the :class:`skore.EstimatorReport` class
# ----------------------------------------------------
#
# Now, we would be interested in getting some insights from our predictive model.
# One way is to use the :class:`skore.EstimatorReport` class. This constructor will
# detect that our estimator is already fitted and will not fit it again.
from skore import EstimatorReport

reporter = EstimatorReport(
    estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)
reporter

# %%
#
# Once the reporter created, we get some information regarding the available tools
# allowing us to get some insights from our specific model on the specific task.
#
# You can get a similar information if you call the :meth:`~skore.EstimatorReport.help`
# method.
reporter.help()

# %%
#
# Be aware that you can access the help for each individual sub-accessor. For instance:
reporter.metrics.help()

# %%
reporter.plot.help()

# %%
#
# Metrics computation with aggressive caching
# -------------------------------------------
#
# At this point, we might be interested to have a first look at the statistical
# performance of our model on the validation set that we provided. We can access it
# by calling any of the metrics displayed above. Since we are greedy, we want to get
# several metrics at once and we will use the
# :meth:`~skore.EstimatorReport.metrics.report_metrics` method.
import time

start = time.time()
metric_report = reporter.metrics.report_metrics(pos_label="allowed")
end = time.time()
metric_report

# %%
print(f"Time taken to compute the metrics: {end - start:.2f} seconds")

# %%
#
# An interesting feature provided by the :class:`skore.EstimatorReport` is the
# the caching mechanism. Indeed, when we have a large enough dataset, computing the
# predictions for a model is not cheap anymore. For instance, on our smallish dataset,
# it took a couple of seconds to compute the metrics. The reporter will cache the
# predictions and if you are interested in computing a metric again or an alternative
# metric that requires the same predictions, it will be faster. Let's check by
# requesting the same metrics report again.

start = time.time()
metric_report = reporter.metrics.report_metrics(pos_label="allowed")
end = time.time()
metric_report

# %%
print(f"Time taken to compute the metrics: {end - start:.2f} seconds")

# %%
# We observe a similar behavior even with metrics that we did not compute before but
# that share the same predictions. So for instance, let's compute the log loss.

start = time.time()
log_loss = reporter.metrics.log_loss()
end = time.time()
log_loss

# %%
print(f"Time taken to compute the log loss: {end - start:.2f} seconds")

# %%
#
# We can show that without initial cache, it would have taken more time to compute
# the log loss.
reporter.clean_cache()

start = time.time()
log_loss = reporter.metrics.log_loss()
end = time.time()
log_loss

# %%
print(f"Time taken to compute the log loss: {end - start:.2f} seconds")

# %%
#
# By default, the metrics are computed on the test set. However, if a training set
# is provided, we can also compute the metrics by specifying the `data_source`
# parameter.
reporter.metrics.log_loss(data_source="train")

# %%
#
# In the case where we are interested in computing the metrics on a completely new set
# of data, we can use the `data_source="X_y"` parameter. In addition, we need to provide
# a `X` and `y` parameters. However, in this case, we cannot safely (FIXME: we might be
# able to do so) track the data provenance and thus not use the cache.

start = time.time()
metric_report = reporter.metrics.report_metrics(
    data_source="X_y", X=X_test, y=y_test, pos_label="allowed"
)
end = time.time()
metric_report

# %%
print(f"Time taken to compute the metrics: {end - start:.2f} seconds")

# %%
#
# Be aware that you can also benefit from the caching mechanism with your own custom
# metrics. We only expect that you define your own metric function to take `y_true`
# and `y_pred` as the first two positional arguments. It can take any other arguments.
# Let's see an example.


def operational_decision_cost(y_true, y_pred, amount):
    mask_true_positive = (y_true == "allowed") & (y_pred == "allowed")
    mask_true_negative = (y_true == "disallowed") & (y_pred == "disallowed")
    mask_false_positive = (y_true == "disallowed") & (y_pred == "allowed")
    mask_false_negative = (y_true == "allowed") & (y_pred == "disallowed")
    # FIXME: we need to make sense of the cost sensitive part with the right naming
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
amount = rng.integers(low=100, high=1000, size=len(y_test))

# %%
#
# Let's make sure that a function called the `predict` method and cached the result.
# We compute the accuracy metric to make sure that the `predict` method is called.
reporter.metrics.accuracy()

# %%
#
# We can now compute the cost of our operational decision.
start = time.time()
cost = reporter.metrics.custom_metric(
    metric_function=operational_decision_cost,
    metric_name="Operational Decision Cost",
    response_method="predict",
    amount=amount,
)
end = time.time()
cost

# %%
print(f"Time taken to compute the cost: {end - start:.2f} seconds")

# %%
#
# Let's now clean the cache and see if it is faster.
reporter.clean_cache()

# %%
start = time.time()
cost = reporter.metrics.custom_metric(
    metric_function=operational_decision_cost,
    metric_name="Operational Decision Cost",
    response_method="predict",
    amount=amount,
)
end = time.time()
cost

# %%
print(f"Time taken to compute the cost: {end - start:.2f} seconds")

# %%
#
# We observe that caching is working as expected. It is really handy because it means
# that you can compute some additional metrics without having to recompute the
# the predictions.
reporter.metrics.report_metrics(
    scoring=["precision", "recall", operational_decision_cost],
    pos_label=pos_label,
    scoring_kwargs={
        "amount": amount,
        "response_method": "predict",
        "metric_name": "Operational Decision Cost",
    },
)

# %%
#
# It could happen that you are interested in providing several custom metrics which
# does not necessarily share the same parameters. In this more complex case, we will
# require you to provide a scorer using the :func:`sklearn.metrics.make_scorer`
# function.
from sklearn.metrics import make_scorer, f1_score

f1_scorer = make_scorer(
    f1_score,
    response_method="predict",
    metric_name="F1 Score",
    pos_label=pos_label,
)
operational_decision_cost_scorer = make_scorer(
    operational_decision_cost,
    response_method="predict",
    metric_name="Operational Decision Cost",
    amount=amount,
)
reporter.metrics.report_metrics(scoring=[f1_scorer, operational_decision_cost_scorer])

# %%
#
# Effortless one-liner plotting
# -----------------------------
#
# The :class:`skore.EstimatorReport` class also provides a plotting interface that
# allows to plot *defacto* the most common plots. As for the the metrics, we only
# provide the meaningful set of plots for the provided estimator.
reporter.plot.help()

# %%
#
# Let's start by plotting the ROC curve for our binary classification task.
display = reporter.plot.roc(pos_label="allowed")

# %%
#
# The plot functionality is built upon the scikit-learn display objects. We return
# those display (slightly modified to improve the UI) in case you want to tweak some
# of the plot properties. You can have quick look at the available attributes and
# methods by calling the `help` method or simply by printing the display.
display

# %%
display.help()

# %%
display.ax_.set_title("Example of a ROC curve")
display.figure_

# %%
#
# Similarly to the metrics, we aggressively use the caching to avoid recomputing the
# predictions of the model. We also cache the plot display object by detection if the
# input parameters are the same as the previous call. Let's demonstrate the kind of
# performance gain we can get.
start = time.time()
# we already trigger the computation of the predictions in a previous call
reporter.plot.roc(pos_label="allowed")
end = time.time()

# %%
print(f"Time taken to compute the ROC curve: {end - start:.2f} seconds")

# %%
#
# Now, let's clean the cache and check if we get a slowdown.
reporter.clean_cache()

# %%
start = time.time()
reporter.plot.roc(pos_label="allowed")
end = time.time()

# %%
print(f"Time taken to compute the ROC curve: {end - start:.2f} seconds")

# %%
# As expected, since we need to recompute the predictions, it takes more time.
