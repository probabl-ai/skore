"""
.. _example_skd007_mdi_cardinality_bias:

SKD007 - MDI feature importance is biased for high-cardinality features
=======================================================================

This example demonstrates the limitations that
:ref:`SKD007 <skd007-mdi-cardinality-bias>` warns against on tree models. Mean
decrease in impurity (MDI) tends to rank high-cardinality categorical or
continuous columns as more important than features with as much signal but
lower cardinality. This is due to the tree building process picking high
cardinality features more often as they offer more split points to choose from.

Mitigations from the :ref:`automated_checks` user guide:

- use permutation importance instead of MDI,
- cross-check MDI with permutation importance or drop-column importance.

We will compare MDI to permutation importance to show that it gives a more
reliable estimate of feature importance. The same contrast is illustrated in
scikit-learn's
`Permutation Importance vs Random Forest Feature Importance (MDI)
<https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html>`_
example.

We fit a :class:`~sklearn.ensemble.RandomForestRegressor` on a 1,500-row
subsample of California housing. The goal is to show the limitations of
impurity based importance and show they do not affect permutation importance
on a test set.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Continuous columns such as ``AveRooms`` and ``AveOccup`` take many distinct
# values: above the 50 % of samples threshold SKD007 uses for high-cardinality
# features.

import numpy as np
from sklearn.model_selection import train_test_split
from skrub.datasets import fetch_california_housing

housing = fetch_california_housing()
X_full, y_full = housing.X, housing.y

X, _, y, _ = train_test_split(
    X_full,
    y_full,
    train_size=1_500,
    random_state=42,
)

# %%
# Let us add two random features that carry no signal about the target: a
# continuous draw from a normal distribution, and a categorical feature with 20
# levels sampled uniformly (stored as integer codes so the forest can split on
# them directly). High-cardinality noise can still receive non-zero MDI, and
# often more MDI than low-cardinality noise, while permutation importance on the
# test set should stay near zero for both.

rng = np.random.default_rng(42)
X["noise_cont"] = rng.normal(size=len(X))
X["noise_cat"] = rng.integers(0, 20, size=len(X))

# %%
# Let us inspect the feature matrix with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
TableReport(y)

# %%
# Counting unique values per column previews which features SKD007 will flag.

X.nunique().sort_values(ascending=False)

# %%
# Trigger SKD007 with a random forest on continuous features
# ==========================================================
#
# A random forest exposes ``feature_importances_`` based on MDI. After fitting,
# let us inspect impurity decrease with skore.

from sklearn.ensemble import RandomForestRegressor
from skore import TrainTestSplit, evaluate

splitter = TrainTestSplit(random_state=42)

report = evaluate(
    RandomForestRegressor(random_state=42),
    X=X,
    y=y,
    splitter=splitter,
)
report

# %%
# SKD007 warns about high-cardinality columns such as ``MedInc`` and
# ``AveOccup``. The synthetic continuous noise column is high-cardinality as
# well, so it belongs in the same tip.

report.checks.summarize(fast_mode=True)

# %%
# Let us plot MDI with features sorted by importance.

import matplotlib.pyplot as plt

mdi_display = report.inspection.impurity_decrease()
_ = mdi_display.plot(sorting_order="descending")

# %%
# The two synthetic noise columns are not near zero under MDI: impurity still
# assigns them mass. ``noise_cont`` in particular is high-cardinality, so the
# forest can keep finding splits on it even though it carries no target signal.

# %%
# Use permutation importance instead of MDI
# =========================================
#
# :meth:`~skore.EstimatorReport.inspection.permutation_importance` shuffles each
# column on the test set and measures the score drop. The result is not biased
# toward high-cardinality split points the way MDI is.

perm_display = report.inspection.permutation_importance(
    seed=42,
    n_repeats=5,
)
_ = perm_display.plot(sorting_order="descending")

# %%
# Under permutation importance the noisy features sit at (or very near) zero:
# shuffling them does not change the test score, so they are not contributing
# to predicting the target.

# %%
# Cross-check MDI with permutation importance
# ===========================================
#
# Let us put the two rankings side by side. Sorting by MDI tends to push
# high-cardinality columns (including ``noise_cont``) upward; permutation
# importance on the test set should keep both synthetic features near the
# bottom even when MDI does not.

mdi = (
    mdi_display.frame(sorting_order="descending")
    .set_index("feature")
    .rename(columns={"importance": "mdi"})
)
perm = (
    perm_display.frame(sorting_order="descending")
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
# The side-by-side bars make the disagreement easier to read: impurity can
# assign mass to ``noise_cont`` (and sometimes more than to ``noise_cat``),
# while permutation importance stays close to zero for both.

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
order = comparison.sort_values("mdi", ascending=True).index

axes[0].barh(order, comparison.loc[order, "mdi"])
axes[0].set_title("MDI (impurity decrease)")
axes[0].set_xlabel("Importance")

axes[1].barh(order, comparison.loc[order, "permutation"])
axes[1].set_title("Permutation importance (test)")
axes[1].set_xlabel("Mean score drop")

fig.tight_layout()
_ = fig

# %%
# Conclusion
# ==========
#
# SKD007 warns that MDI feature importance favors high-cardinality inputs such
# as ``AveOccup`` and can inflate the role of irrelevant high-cardinality noise.
# In this walkthrough, permutation importance gave a more reliable picture of
# which features actually move test scores. When importance is a decision
# factor, we prefer permutation (or drop-column tests) over impurity alone.
