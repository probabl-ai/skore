"""
===============
Cache mechanism
===============

In this example, we dive into some internal details of :class:`~skore.EstimatorReport`
and :class:`~skore.CrossValidationReport` regarding their ability to cache critical
information to speed-up long computations.
"""

# %%
#
# We start by loading a dataset using the `skrub` library. The task in this case is
# to predict whether a company made a payment to a physician.
from skrub.datasets import fetch_open_payments

dataset = fetch_open_payments()
df = dataset.X
y = dataset.y

# %%
from skrub import TableReport

TableReport(df)

# %%
#
# We observe that the dataset contains more than 70,000 records and only contains
# categorical features, some where categories are not well-defined. To handle this
# issue, we use `skrub` to define a baseline predictive model.
from skrub import tabular_learner

model = tabular_learner("classifier")
model

# %%
#
# This model handles numerical, categorical, date and time, and missing values out of
# the box. We now train the model on part of the dataset.
from skore import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=42)

# %%
#
# Now, we focus on the :class:`~skore.EstimatorReport` class and more specifically
# its ability to cache some information to speed-up long computations related to the
# prediction of the model. Let's first train the model by passing the training data
# to the :class:`~skore.EstimatorReport` class.
from skore import EstimatorReport

reporter = EstimatorReport(
    model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
)
reporter.help()

# %%
#
# Now, that the model is trained, we can use our `reporter`. For instance, we can
# start by computing on the metric on the test set.
import time

start = time.time()
result = reporter.metrics.accuracy()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# To compute the accuracy, the `reporter` is calling the following scikit-learn
# code (or really similar code) under the hood.
from sklearn.metrics import accuracy_score

start = time.time()
result = accuracy_score(reporter.y_test, reporter.estimator_.predict(reporter.X_test))
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# We observe that the time taken to compute the accuracy with both approaches is
# close which make sense. Let's call again the computation of the accuracy with the
# `reporter` object.
start = time.time()
result = reporter.metrics.accuracy()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# Surprisingly, the second call to the `reporter.metrics.accuracy()` method was
# instantenaous. The reason is that the `reporter` stored internal information allowing
# to avoid re-computing the accuracy from scratch. We can check the internal
# `_cache` attribute of the `reporter` to see what is stored.
reporter._cache

# %%
#
# We observe 2 keys in the `_cache` attribute. We can try to decrypt the content of
# each key.
list(reporter._cache.keys())[0]

# %%
#
# While the first number in the tuple, might be cryptic, the second and third values
# are easier to understand: the second value corresponds to the type of predictions
# that we needed to compute the accuracy. The third value is associated with the source
# of data for which we need the predictions. Finally, the first value is a hash related
# to when the `reporter` was created.
reporter._hash

# %%
#
# So from, the information in the cache, we expect that if we compute a new metric
# requiring the same type of predictions, then it will be loaded from the cache.
# Only the cost of to compute the metric will be paid. Let's take the example of the
# precision metric.
start = time.time()
result = reporter.metrics.precision()
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
#
# It took less time than computing the accuracy but we still have to pay the cost of
# computing the metric. Now, we can check what has been added to the cache.
reporter._cache

# %%
#
# We observe that the cache now contains information related to the precision metric.
# So we now see that the `reporter` is adding information if necessary at each call
# of a metric. We expose the :meth:`~skore.EstimatorReport.cache_predictions` to
# precompute all predictions at once instead of doing it on-the-fly. This method also
# allows parallel computation of the predictions.
reporter.cache_predictions(n_jobs=2)

# %%
#
# Now, so now we stored all type of possible predictions for the given estimator and
# all provided data sources at initialization (i.e. `train` and `test` sets). It means
# that we are not going to pay the cost of computing the predictions at each call of
# any metric. Let's check the `log_loss` metric on the train set.
start = time.time()
result = reporter.metrics.log_loss(data_source="train")
end = time.time()
result

# %%
print(f"Time taken: {end - start:.2f} seconds")

# %%
