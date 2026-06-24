"""
Tracking all the data processing
================================

To track all operations and be able to apply the fitted estimator to unseen
data, we need to include all the data wrangling in the estimator used for our
skore report. In very simple cases this can be done with a scikit-learn
Pipeline. When we have transformations not supported by the Pipeline (such as
transformations that change the number of rows, or that involve multiple tables
such as joins), skore allows us to use a skrub DataOp instead.

In this example we consider a dataset that is simple, but still requires some
data wrangling (encoding, aggregation and joining) which could not be performed
in a regular scikit-learn estimator.

To track those operations, we use a skrub DataOp, which can perform richer
transformations than normal estimators, and also has built-in support from
skore.

The dataset contains a list of online transactions (each corresponds to a cart,
or "basket"), each linked to one or more products for which we have a description.
The task is to predict which involved credit fraud.
"""

# %%
# We start by defining our data-processing pipeline. Note that it contains
# operations, such as aggregating and joining the product information after
# vectorizing the text it contains, that would not be possible in a normal
# estimator.

# %%
import skore
import skrub
from sklearn.ensemble import HistGradientBoostingClassifier

dataset = skrub.datasets.fetch_credit_fraud(split="all")

products = skrub.var("products", dataset.products)
baskets = skrub.var("baskets", dataset.baskets)

basket_ids = baskets[["ID"]].skb.mark_as_X()
fraud_flags = baskets["fraud_flag"].skb.mark_as_y()


def filter_products(products, basket_ids):
    return products[products["basket_ID"].isin(basket_ids["ID"])]


vectorized_products = products.skb.apply_func(filter_products, basket_ids).skb.apply(
    skrub.TableVectorizer(), exclude_cols="basket_ID"
)


def join_product_info(basket_ids, vectorized_products):
    return basket_ids.merge(
        vectorized_products.groupby("basket_ID").agg("mean").reset_index(),
        left_on="ID",
        right_on="basket_ID",
    ).drop(columns=["ID", "basket_ID"])


pred = basket_ids.skb.apply_func(join_product_info, vectorized_products).skb.apply(
    HistGradientBoostingClassifier(), y=fraud_flags
)

# This would generate a report with previews of intermediate results & fitted
# estimators:
#
# pred.skb.full_report()

pred

# %%
# Above we see a preview on the whole dataset. Click the "show graph" toggle to
# see a drawing of the pipeline we have built.
#
# Just like a normal estimator, a skrub DataOp can be used with skore reports.
# We can either pass separately a SkrubLearner and training and testing data,
# or pass our DataOp with the data it already contains and rely on the default
# train/test split:

# %%
report = skore.EstimatorReport(pred, pos_label=1)
report.metrics.roc_auc()

# %%
_ = report.metrics.precision_recall().plot()

# %%
# Note that the preprocessing operations are captured in the skrub DataOp,
# hence in our report -- so we can replay them later on unseen data.

# %%
report.estimator_.data_op
