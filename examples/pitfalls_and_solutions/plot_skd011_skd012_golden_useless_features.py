"""
.. _example_skd011_golden_feature_skd012_useless_features:

SKD011 & SKD012 - Golden feature and useless features
=====================================================

This example walks through mitigations when checks
:ref:`SKD011 <skd011-golden-feature>` and :ref:`SKD012 <skd012-useless-features>`
fire together: a common pattern when a leaky column carries almost all signal
and remaining inputs look negligible by comparison.

Mitigations from the :ref:`automated_checks` user guide, in the order we try
them here:

**SKD011 - golden feature**

- audit the suspect feature for leakage,
- compare predictive performance with and without the feature,
- collect or engineer additional features so the model is less dependent on
  a single one.

**SKD012 - useless features**

- review the flagged features and consider dropping them,
- refit on a reduced feature set and verify performance is preserved,
- if a flagged feature should matter, investigate encoding (here: feature
  engineering before pruning).

We use the medical charge dataset with leakage columns retained in the
with-leakage table. The goal is to audit suspect aggregates, remove leakage,
re-encode concentrated DRG signal without duplicating it, and only then prune
genuinely weak columns.
"""

# %%
# Load the medical charge dataset
# ===============================
#
# The target is ``Average_Total_Payments``. Columns such as
# ``Average_Medicare_Payments`` are billing aggregates from the same process;
# they are not available before the target is known in a real deployment. We
# keep them in the with-leakage table to show what happens when leakage slips
# through validation.
#
# SKD011 and SKD012 often fire on the same report: a golden feature alone
# matches full-model scores, so other columns look useless even though they can
# matter once the golden column is removed. Audit leakage (SKD011) before
# dropping features flagged by SKD012.
#
# Both checks are slow (per-feature refits / permutation importance). Run
# full :meth:`~skore.EstimatorReport.checks.summarize` on the trigger and on the
# cleaned report so you can see both checks clear.

from skrub.datasets import fetch_medical_charge

dataset = fetch_medical_charge()
X, y = dataset.X, dataset.y

# %%
# Inspect predictors and target with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
TableReport(y)

# %%
# Drop provider identifiers for modelling, subsample 3,000 rows, and build two
# tables: one with leakage columns and one without.

id_cols = [
    "Provider_Zip_Code",
    "Provider_Id",
    "Provider_Name",
    "Provider_Street_Address",
]
leakage_cols = ["Average_Covered_Charges", "Average_Medicare_Payments"]

X_with_leakage = X.drop(columns=id_cols).sample(3_000, random_state=42)
y_sub = y.loc[X_with_leakage.index]
X_without_leakage = X_with_leakage.drop(columns=leakage_cols)

# %%
from skore import TrainTestSplit

splitter = TrainTestSplit(random_state=42, test_size=0.2)

# %%
# Trigger SKD011 and SKD012 - with leakage
# ========================================

from skore import evaluate
from skrub import tabular_pipeline

report_with_leakage = evaluate(
    tabular_pipeline("regressor"),
    X=X_with_leakage,
    y=y_sub,
    splitter=splitter,
)

# %%
# Expect tips for SKD011 on ``Average_Medicare_Payments`` and SKD012 on columns
# overshadowed by that leakage feature. Near-perfect scores plus a golden
# feature are a leakage smoke signal so pause before treating SKD012 flags as a list
# of columns to drop.

report_with_leakage.checks.summarize()

# %%
report_with_leakage.metrics.summarize(data_source="both").frame()

# %%
# SKD011 - audit the suspect feature for leakage
# ==============================================
#
# We correlate leakage columns with the target and inspect cardinality. A column
# that is almost collinear with the label and available only after billing
# closes should not ship to production.

leak_audit = (
    X_with_leakage[leakage_cols]
    .corrwith(y_sub)
    .rename("corr_with_target")
    .to_frame()
    .assign(unique_values=X_with_leakage[leakage_cols].nunique())
    .sort_values("corr_with_target", key=abs, ascending=False)
)
leak_audit

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(
    X_with_leakage["Average_Medicare_Payments"],
    y_sub,
    alpha=0.4,
    s=10,
)
ax.set(
    xlabel="Average_Medicare_Payments",
    ylabel=y_sub.name,
    title="Suspect feature vs target",
)
_ = fig

# %%
# A tight scatter against the target confirms the column describes the same
# outcome you are trying to forecast, not a legitimate precursor.
#
# SKD011 - compare with and without leakage
# =========================================
#
# :func:`~skore.compare` contrasts metrics on the leaky and clean tables. Test
# scores should drop once legitimate predictors must carry the signal alone.

from skore import compare

report_without_leakage = evaluate(
    tabular_pipeline("regressor"),
    X=X_without_leakage,
    y=y_sub,
    splitter=splitter,
)

comparison = compare(
    {
        "with_leakage": report_with_leakage,
        "without_leakage": report_without_leakage,
    }
)
comparison.metrics.summarize(data_source="both").frame()

# %%
# Without leakage, SKD011 should clear. SKD012 may still flag weak columns, but
# those flags are no longer caused by a golden feature. As we see now there is only
# one weak column.

report_without_leakage.checks.summarize()

# %%
# After leakage is gone, ``DRG_Definition`` still carries most of the signal.

perm_without_leakage = report_without_leakage.inspection.permutation_importance(
    seed=42,
    n_repeats=5,
)
perm_without_leakage.frame()

# %%
_ = perm_without_leakage.plot()

# %%
# It seems like that most of the signal lives in ``DRG_Definition``;
# so next we re-encode it.

# %%
# SKD011 - engineer legitimate features
# =====================================
#
# We *decompose* ``DRG_Definition`` into parts the model can use separately:
#
# - ``DRG_Code`` (string): the categorical procedure id, kept as text so
#   ``tabular_pipeline`` treats it as high-cardinality, not a float scale;
# - ``has_MCC`` / ``has_CC``: severity bits that many DRGs share.
#
# Then we drop the original free-text column (id and definition are the same, so
# keeping both would duplicate the same identity).
#
# We wrap the transform in a :class:`~sklearn.preprocessing.FunctionTransformer`
# and put it at the start of the estimator pipeline so the same engineering
# runs on train and test (and cannot leak fit-time statistics).


def engineer_features(X):
    X = X.copy()
    if "DRG_Definition" in X.columns:
        X["DRG_Code"] = X["DRG_Definition"].str.extract(r"^(\d+)", expand=False)
        drg = X["DRG_Definition"].str.upper()
        X["has_MCC"] = drg.str.contains("W MCC", regex=False).astype(int)
        X["has_CC"] = (
            drg.str.contains("W CC", regex=False)
            & ~drg.str.contains("W MCC", regex=False)
        ).astype(int)
        X = X.drop(columns=["DRG_Definition"])
    return X


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

model_engineered = make_pipeline(
    FunctionTransformer(engineer_features),
    tabular_pipeline("regressor"),
)

report_engineered = evaluate(
    model_engineered,
    X=X_without_leakage,
    y=y_sub,
    splitter=splitter,
)

report_engineered

# %%
# Scores may move because the model no longer sees the raw free-text field,
# only the decomposed ``DRG_Code`` and severity flags inside the pipeline.

comparison_engineered = compare(
    {
        "without_leakage": report_without_leakage,
        "with_feature_engineering": report_engineered,
    }
)
comparison_engineered.metrics.summarize(data_source="both").frame()

# %%
report_engineered.checks.summarize(fast_mode=True)

# %%
# SKD012 - prune weak features inside the pipeline
# ================================================
#
# On the *leaky* report, SKD012 often flagged geography and discharges because
# Medicare payments already explained the target; those were overshadowed, not
# a drop list. After cleaning and engineering, we prune weak *vectorized*
# signal with :class:`~sklearn.feature_selection.SelectFromModel` so selection
# is fit on the training fold only (same rule as any other pipeline step).
#
# We keep the same :func:`~sklearn.preprocessing.FunctionTransformer` first,
# vectorize with :class:`~skrub.TableVectorizer`, select with a forest, then
# fit the final regressor on the reduced representation.

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from skrub import TableVectorizer

model_reduced = make_pipeline(
    FunctionTransformer(engineer_features),
    TableVectorizer(),
    SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=42),
        threshold="median",
    ),
    HistGradientBoostingRegressor(random_state=42),
)
model_reduced

# %%
report_reduced = evaluate(
    model_reduced,
    X=X_without_leakage,
    y=y_sub,
    splitter=splitter,
)

report_reduced

# %%
comparison_reduced = compare(
    {
        "with_feature_engineering": report_engineered,
        "reduced_features": report_reduced,
    }
)
comparison_reduced.metrics.summarize(data_source="test").frame()

# %%
report_reduced.checks.summarize(fast_mode=True)

# %%
# Conclusion
# ==========
#
# SKD011 and SKD012 often appear together when one leaky column dominates.
# Audit that column, compare with-and-without it, then re-encode concentrated
# signal without duplication (here: string ``DRG_Code`` plus severity flags
# instead of free-text ``DRG_Definition``, inside the estimator pipeline). Only
# then prune columns that remain weak on the clean table, for example with
# :class:`~sklearn.feature_selection.SelectFromModel`. Do not treat SKD012
# flags on a leaky report as a drop list.
