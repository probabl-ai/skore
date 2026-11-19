"""
.. _example_skrub_permutation_importance:

=======================================================
Permutation importance on the inputs of a skrub learner
=======================================================

Permutation importance is usually computed on the features that are fed to the
predictor. When the features are the result of a complex data preparation that is
part of a skrub :class:`~skrub.DataOp`, the engineered columns are often hard to
interpret. In this example we compute the importance of the columns of an
intermediate table, *before* it is vectorized and aggregated, which is more
meaningful to the user: does the prediction rely on the brand of the products or
on their price?

We use the credit fraud dataset, in which each basket contains several items.
"""

# %%
# Build the skrub learner
# =======================
#
# The ``baskets`` table has one row per order, with the target ``fraud_flag``. The
# ``products`` table has one row per item purchased in a basket. We vectorize the
# items, aggregate them by basket and join them to the baskets before fitting
# a gradient boosting classifier.
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

dataset = skrub.datasets.fetch_credit_fraud(split="train")

data_source = skrub.var(
    "data_source", {"orders": dataset.baskets, "items": dataset.products}
)


def load_orders(data_source, order_ids=None):
    if order_ids is None:
        return data_source["orders"]
    return data_source["orders"].set_index("ID").loc[order_ids].reset_index(drop=False)


query = skrub.var("order_ids", dataset.baskets["ID"])
orders = data_source.skb.apply_func(load_orders, query)
order_ids = orders[["ID"]].skb.mark_as_X()
fraud_flags = orders["fraud_flag"].skb.mark_as_y()


def load_items(order_ids, data_source):
    items = data_source["items"]
    return items[items["basket_ID"].isin(order_ids["ID"])]


# We name the raw items table, so that we can inspect it later.
items = order_ids.skb.apply_func(load_items, data_source).skb.set_name("items")
vectorized_items = items.skb.apply(skrub.TableVectorizer(), exclude_cols="basket_ID")


def join_items(orders, vectorized_items):
    return orders.merge(
        vectorized_items.groupby("basket_ID").agg("mean").reset_index(),
        left_on="ID",
        right_on="basket_ID",
    ).drop(columns=["ID", "basket_ID"])


features = order_ids.skb.apply_func(join_items, vectorized_items)
pred = features.skb.apply(HistGradientBoostingClassifier(), y=fraud_flags)

# %%
# We split the data, and fit the learner on the training set.
split = pred.skb.train_test_split()
learner = pred.skb.make_learner().fit(split["train"])

# %%
# Create a report
# ===============
#
# We pass the fitted learner and the test data to an
# :class:`~skore.EstimatorReport`.
import skore

report = skore.EstimatorReport(learner, test_data=split["test"])

# %%
# Permutation importance of the raw items
# =======================================
#
# For a skrub learner, ``at_step`` is the name (or the ID) of the node of the
# DataOp whose output we want to shuffle. Here we shuffle the columns of the
# ``"items"`` table, independently of each other. Note that this table is not
# row-aligned with the target, which has one entry per basket rather than one per
# item: it is not a problem for the permutation importance, but it means that
# subsampling with ``max_samples`` is not supported.
display = report.inspection.permutation_importance(
    at_step="items", metric="roc_auc", n_repeats=5, seed=0
)
display.frame()

# %%
display.plot(sorting_order="descending")

# %%
# The importance of ``basket_ID`` is a useful sanity check: shuffling this column
# breaks the link between items and baskets, so the model can no longer do better
# than chance. The other columns describe the products: we can see how much the
# model relies on their model, make, price, etc.
#
# By default (``at_step=None``) the importance is computed on the inputs of the
# learner, here the order IDs. This only tells us that the model needs them to
# retrieve the items of each order, and is not informative.
