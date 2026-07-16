"""
.. _example_skd007_mdi_cardinality_bias:

SKD007 — MDI feature importance is biased for high-cardinality features
=======================================================================

This example demonstrates mitigations when check
:ref:`SKD007 <skd007-mdi-cardinality-bias>` tips on tree models. Mean decrease
in impurity (MDI) tends to rank high-cardinality continuous columns as more
important because they offer more split points — even when held-out
permutation scores disagree.

Mitigations from the :ref:`automated_checks` user guide:

- use permutation importance instead of MDI,
- cross-check MDI with permutation importance or drop-column importance.

We fit a :class:`~sklearn.ensemble.RandomForestRegressor` on a 1,500-row
subsample of California housing. The goal is to read importance through
permutation on the test set rather than impurity alone.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Continuous columns such as ``AveRooms`` and ``AveOccup`` take many distinct
# values — above the 50 % of samples threshold SKD007 uses for
# high-cardinality features.

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)
X_full = housing.frame.drop(columns=["MedHouseVal"])
y_full = housing.frame["MedHouseVal"]

X, _, y, _ = train_test_split(
    X_full,
    y_full,
    random_state=42,
)

# %%
# Inspect the feature matrix with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
TableReport(y)

# %%
# Counting unique values per column previews which features SKD007 will flag.

X.nunique().sort_values(ascending=False)

# %%
# Trigger SKD007 — random forest on continuous features
# =====================================================
#
# A random forest exposes ``feature_importances_`` based on MDI. After fitting,
# inspect impurity decrease with skore.

from sklearn.ensemble import RandomForestRegressor
from skore import TrainTestSplit, evaluate

splitter = TrainTestSplit(random_state=42)

report = evaluate(
    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD007 should tip on high-cardinality columns such as ``MedInc`` and
# ``AveOccup``.

report.checks.summarize(fast_mode=True)

# %%
report.inspection.impurity_decrease().frame().sort_values("importance", ascending=False)

# %%
# Use permutation importance instead of MDI
# =========================================
#
# :meth:`~skore.EstimatorReport.inspection.permutation_importance` shuffles each
# column on the test set and measures the score drop. The result is not biased
# toward high-cardinality split points the way MDI is.

import matplotlib.pyplot as plt

perm_display = report.inspection.permutation_importance(
    seed=42,
    n_repeats=5,
)
perm_display.frame().sort_values("value_mean", ascending=False)

# %%
fig = perm_display.plot()
plt.show()

# %%
# Cross-check MDI with permutation importance
# ===========================================
#
# Join MDI, permutation importance, and unique-value counts. Sorting by MDI
# puts high-cardinality columns near the top; compare that order with the
# permutation ranking to see where impurity overstates importance.

mdi = (
    report.inspection.impurity_decrease()
    .frame()
    .set_index("feature")
    .rename(columns={"importance": "mdi"})
)
perm = (
    perm_display.frame()
    .set_index("feature")[["value_mean"]]
    .rename(columns={"value_mean": "permutation"})
)
nunique = X.nunique().rename("nunique")

comparison = (
    mdi.join(perm)
    .join(nunique)
    .assign(
        mdi_rank=lambda df: df["mdi"].rank(ascending=False).astype(int),
        perm_rank=lambda df: df["permutation"].rank(ascending=False).astype(int),
    )
    .sort_values("mdi", ascending=False)
)
comparison

# %%
# Conclusion
# ==========
#
# SKD007 warns that MDI feature importance favors high-cardinality inputs such as
# AveOccup. In this walkthrough, permutation importance and a direct MDI comparison
# offered a more reliable picture of which features actually move test scores more.
# Prefer permutation (or drop-column tests) when importance is a decision factor.
