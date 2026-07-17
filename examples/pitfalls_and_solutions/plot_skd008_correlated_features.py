"""
.. _example_skd008_correlated_features:

SKD008 — Highly correlated input features
=========================================

This example walks through mitigations when check
:ref:`SKD008 <skd008-correlated-features>` fires because numeric columns are
nearly redundant. The check computes pairwise Spearman correlation on training
inputs and flags pairs with :math:`|\rho| > 0.9`.

Mitigations from the :ref:`automated_checks` user guide:

- remove or combine redundant features,
- use regularization (Lasso, ElasticNet),
- group correlated features before inspecting importance.

We use the breast cancer Wisconsin dataset, where radius, perimeter, and area
measurements are almost linearly related. The goal is to simplify the feature
table without losing discriminative signal.
"""

# %%
# Load the breast cancer Wisconsin dataset
# ========================================
#
# Thirty numeric nucleus measurements describe cell size, shape, and texture.
# Many pairs exceed the 0.9 Spearman threshold. Scores are already high on this
# table; the point is to surface collinearity, not to chase marginal accuracy
# gains.

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer(as_frame=True)
X = cancer.frame.drop(columns=["target"])
y = cancer.target

# %%
# :class:`~skrub.TableReport` shows the dense numeric design matrix.

from skrub import TableReport

TableReport(X)

# %%
# The binary target is moderately imbalanced but easy to separate — collinearity
# is the focus, not class balance.

TableReport(y)

# %%
# Stratified :class:`~skore.TrainTestSplit` keeps both classes in train and
# test. Named column groups below support the grouping mitigation later.

from skore import TrainTestSplit

splitter = TrainTestSplit(random_state=42, stratify=y)

# %%
# Trigger SKD008 — full feature set
# =================================
#
# A gradient boosting classifier tolerates correlated inputs but SKD008 still
# inspects the training matrix. Fit on the full table, then summarize checks.

from sklearn.ensemble import HistGradientBoostingClassifier
from skore import evaluate

report = evaluate(
    HistGradientBoostingClassifier(random_state=42),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD008 give the number of highly correlated feature pairs on the training fold.

report.checks.summarize(fast_mode=True)

# %%
# Investigate correlated pairs on train data
# ==========================================
#
# SKD008 runs on **train** inputs. Lets check the correlated pairs on the train data.
# manually to see how many there are.

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
pairs.head(28)

# %%
# Remove redundant features
# =========================
#
# Drop perimeter and area within each size block, remove entire error and worst
# blocks, and discard columns highly correlated with ``mean concavity``. Refit
# on the reduced table and re-run checks.

redundant = [
    "mean perimeter",
    "mean area",
    "mean concave points",
    "mean compactness",
] + [c for c in X.columns if c.startswith("worst") or c.endswith("error")]
X_dropped = X.drop(columns=redundant)

report_dropped = evaluate(
    HistGradientBoostingClassifier(random_state=42),
    X=X_dropped,
    y=y,
    splitter=splitter,
)

# %%
# SKD008 should be absent once redundant pairs are removed from training inputs.

report_dropped.checks.summarize(fast_mode=True)

# %%
# Use regularization (L1 logistic regression)
# ===========================================
#
# L1 penalty can shrink one member of a correlated pair toward zero, which
# helps interpretation at fit time. SKD008 inspects **inputs**, though — the
# check still fires on the original ``X`` because column redundancy remains.

from sklearn.linear_model import LogisticRegression

report_lasso = evaluate(
    LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.1,
        max_iter=10_000,
        random_state=42,
    ),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# SKD008 still fires — the feature table is unchanged.

report_lasso.checks.summarize(fast_mode=True)

# %%
# Here we see the coefficients of the logistic regression model and all but 6 of them
# have been reduced to 0.
report_lasso.inspection.coefficients().frame()

# %%
# Group correlated features
# =========================
#
# Collapse each redundant block into one summary column — mean tumor size,
# texture, smoothness, and so on — before fitting. Fewer, uncorrelated
# summaries often clear SKD008 while preserving domain structure.

SIZE_ALL = [
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
        "tumor_size": X[SIZE_ALL].mean(axis=1),
        "tumor_texture": X[TEXTURE].mean(axis=1),
        "tumor_smoothness": X[SMOOTHNESS].mean(axis=1),
        "tumor_symmetry": X[SYMMETRY].mean(axis=1),
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
# SKD008 should be absent on the six summary columns.

report_grouped.checks.summarize(fast_mode=True)

# %%
# Compare strategies
# ==================
#
# :func:`~skore.compare` contrasts test metrics for the full, dropped, and
# grouped feature tables on the same stratified split. This dataset is
# really simple so the metrics are not that different between the three strategies
# uncommonly high.

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
# Conclusion
# ==========
#
# SKD008 highlights redundant numeric features that complicate interpretation.
# Here, checking the correlation on training data manually guided both aggressive dropping
# and structured grouping; L1 regularization helped coefficients but did not
# clear the check. Choose dropping or aggregation based on which how you want to
# modify the feature table.

# %%
