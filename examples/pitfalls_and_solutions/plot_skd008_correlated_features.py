"""
.. _example_skd008_correlated_features:

SKD008 - Highly correlated input features
=========================================

This example walks through mitigations when check
:ref:`SKD008 <skd008-correlated-features>` fires because numeric columns are
nearly redundant. The check computes pairwise Spearman correlation on training
inputs and flags pairs with :math:`|ρ| > 0.9`.

Mitigations from the :ref:`automated_checks` user guide:

- remove or combine redundant features,
- use regularization (Lasso, ElasticNet),
- group correlated features before inspecting importance.

We use the breast cancer Wisconsin dataset, where radius, perimeter, and area
measurements are almost linearly related. The goal is to simplify the feature
table without losing signal.
"""

# %%
# Load the breast cancer Wisconsin dataset
# ========================================
#
# The dataset describes cell nuclei with 30 numeric features pertaining to
# cell size, shape, and texture. Many of these features are correlated;
# the check should help detect this.

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(as_frame=True, return_X_y=True)

# %%
# Thanks to the :class:`~skrub.TableReport` "Associations" tab we can already
# see many correlated feature pairs.

from skrub import TableReport

TableReport(X)

# %%
# The target is moderately imbalanced but easy to separate; anyway, our concern
# in this example is collinearity of features.

TableReport(y)

# %%
# Let us use a stratified :class:`~skore.TrainTestSplit` so both classes appear
# in train and test. Named column groups below support the combine mitigation
# later.

from skore import TrainTestSplit

splitter = TrainTestSplit(random_state=42, stratify=y)

# %%
# Trigger SKD008: full feature set
# ================================
#
# A gradient boosting classifier tolerates correlated inputs, but SKD008 still
# inspects the training matrix. Let us fit on the full table, then summarize
# checks.

from sklearn.ensemble import HistGradientBoostingClassifier
from skore import evaluate

report = evaluate(
    HistGradientBoostingClassifier(random_state=42),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD008 gives the number of highly correlated feature pairs on the training
# fold.

report.checks.summarize(fast_mode=True)

# %%
# Investigate correlated pairs on train data
# ==========================================
#
# SKD008 runs on **train** inputs. Let us check the correlated pairs on the
# train data manually to see how many there are.

import numpy as np
import pandas as pd

X_train = pd.DataFrame(report.X_train, columns=X.columns)

corr = X_train.corr(method="spearman").abs()
pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
    .rename("abs_spearman")
)
# The first 27 pairs sit above the 0.9 threshold; after that, |ρ| drops below
# it. We show a few extra rows for context.
pairs.head(28)

# %%
# Remove redundant features
# =========================
#
# One way to satisfy the check is to drop some of the correlated features.
# Let us drop:
#
# - perimeter and area within each size block since they are correlated with
#   radius,
# - the "error" and "worst" features,
# - the features highly correlated with ``mean concavity``.

X_dropped = X.drop(
    columns=[
        "mean perimeter",
        "mean area",
        "mean concave points",
        "mean compactness",
    ]
    + [c for c in X.columns if "worst" in c or "error" in c]
)

report_dropped = evaluate(
    HistGradientBoostingClassifier(random_state=42),
    X=X_dropped,
    y=y,
    splitter=splitter,
)

# %%
# SKD008 no longer fires.

report_dropped.checks.summarize(fast_mode=True)

# %%
# Combine correlated features
# ===========================
#
# Another way to keep the check from firing is to combine redundant features
# together, e.g. by taking their mean. This is similar to removing correlated
# features, but without completely removing the information available in the
# removed features.

SIZE = [
    "mean radius",
    "mean perimeter",
    "mean area",
    "radius error",
    "perimeter error",
    "area error",
    "worst radius",
    "worst perimeter",
    "worst area",
]
TEXTURE = ["mean texture", "texture error", "worst texture"]
SMOOTHNESS = ["mean smoothness", "smoothness error", "worst smoothness"]
SYMMETRY = ["mean symmetry", "symmetry error", "worst symmetry"]
FRACTAL = [
    "mean fractal dimension",
    "fractal dimension error",
    "worst fractal dimension",
]
SHAPE_IRREGULARITY = [
    "mean compactness",
    "compactness error",
    "worst compactness",
    "mean concavity",
    "concavity error",
    "worst concavity",
    "mean concave points",
    "concave points error",
    "worst concave points",
]

X_grouped = pd.DataFrame(
    {
        "size": X[SIZE].mean(axis=1),
        "texture": X[TEXTURE].mean(axis=1),
        "smoothness": X[SMOOTHNESS].mean(axis=1),
        "symmetry": X[SYMMETRY].mean(axis=1),
        "fractal_dimension": X[FRACTAL].mean(axis=1),
        "shape_irregularity": X[SHAPE_IRREGULARITY].mean(axis=1),
    }
)

# %%
report_grouped = evaluate(
    HistGradientBoostingClassifier(random_state=42),
    X=X_grouped,
    y=y,
    splitter=splitter,
)

# %%
# SKD008 no longer fires.

report_grouped.checks.summarize(fast_mode=True)

# %%
# Compare strategies
# ==================
#
# :func:`~skore.compare` contrasts test metrics for the full, dropped, and
# combined feature tables on the same stratified split. This dataset is
# really simple so the metrics are not that different between the three
# strategies.

from skore import compare

comparison = compare(
    {
        "full_features": report,
        "dropped_redundant": report_dropped,
        "grouped_features": report_grouped,
    }
)
comparison.metrics.summarize(data_source="both").frame(favorability=True)

# %%
# Note: regularization does not clear SKD008
# ==========================================
#
# The L1 regularization can shrink one member of a correlated pair toward zero.
# However, SKD008 only inspects the input data, so changing the estimator does
# not impact the check result.

from sklearn.linear_model import LogisticRegression

report_lasso = evaluate(
    LogisticRegression(penalty="l1", solver="liblinear", random_state=42),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD008 still fires.

report_lasso.checks.summarize(fast_mode=True)

# %%
# Conclusion
# ==========
#
# SKD008 highlights redundant numeric features that may cause the fitting to
# fail or complicate interpretation. Here, checking the correlation on training
# data manually guided both aggressive dropping and structured combining; L1
# regularization can help coefficients but does not clear the check. Choose
# dropping or aggregation based on how you want to modify the feature table.
