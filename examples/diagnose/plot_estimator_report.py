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
positive_class, negative_class = "allowed", "disallowed"

# %%
# Before to train a predictive model, we need to split our dataset into a training
# and a validation set.
from skore import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df, y, random_state=42)

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
# Now, we would be interested in getting some insights from our predictive model.
# One way is to use the :class:`skore.EstimatorReport` class. Since our model is already
# trainer, we can call the :meth:`~skore.EstimatorReport.from_fitted_estimator` and pass
# the dataset to validate our model.
from skore import EstimatorReport

reporter = EstimatorReport.from_fitted_estimator(estimator, X=X_val, y=y_val)
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
# At this point, we might be interested to have a first look at the statistical
# performance of our model on the validation set that we provided. We can access it
# by calling any of the metrics displayed above. Since we are greedy, we want to get
# several metrics at once and we will use the
# :meth:`~skore.EstimatorReport.metrics.report_metrics` method.
import time

start = time.time()
metric_report = reporter.metrics.report_metrics(positive_class="allowed")
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
metric_report = reporter.metrics.report_metrics(positive_class="allowed")
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
# Another feature that is handy is to be able to compute the same statistics on a
# completely new set of data. The metrics above accept a `X` and `y` parameters that
# allow to pass a new set of data to compute the metrics on. However, in this case,
# we cannot safely (FIXME: we might be able to do so) track the data provenance and
# thus not use the cache.

start = time.time()
metric_report = reporter.metrics.report_metrics(
    X=X_val, y=y_val, positive_class="allowed"
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
amount = rng.integers(low=100, high=1000, size=len(y_val))

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
    positive_class=positive_class,
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
    pos_label=positive_class,
)
operational_decision_cost_scorer = make_scorer(
    operational_decision_cost,
    response_method="predict",
    metric_name="Operational Decision Cost",
    amount=amount,
)
reporter.metrics.report_metrics(scoring=[f1_scorer, operational_decision_cost_scorer])

# %%
