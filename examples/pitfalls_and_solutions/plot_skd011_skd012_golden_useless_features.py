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

- audit the suspect feature for leakage (is it derived from the target or
  from data that would not be available at inference time?),
- decide whether to keep or drop it,
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
# Let's load the `medical_charge` dataset, which we will use to predict the average
# cost of an inpatient stay from the hospital's location and the type of medical
# procedure.

from skrub.datasets import fetch_medical_charge

dataset = fetch_medical_charge()
X_full, y_full = dataset.X, dataset.y

# %%
# Then we can inspect predictors and target with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X_full)

# %%
TableReport(y_full)

# %%
# Next, we drop provider identifiers for modelling and subsample 3,000 rows.

id_cols = [
    "Provider_Zip_Code",
    "Provider_Id",
    "Provider_Name",
    "Provider_Street_Address",
]

X = X_full.drop(columns=id_cols).sample(3_000, random_state=42).reset_index(drop=True)
y = y_full.sample(3_000, random_state=42).reset_index(drop=True)

# %%
# We now create a splitter, vectorizer, and regressor that we reuse throughout
# the example. :func:`~skore.evaluate` clones them before fitting.
#
# High-cardinality strings are encoded with :class:`~skrub.StringEncoder` (TF-IDF
# then randomized SVD); we seed it so reruns stay comparable.

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from skore import TrainTestSplit
from skrub import StringEncoder, TableVectorizer

splitter = TrainTestSplit(random_state=42, test_size=0.2)
vectorizer = TableVectorizer(high_cardinality=StringEncoder(random_state=42))
regressor = HistGradientBoostingRegressor(random_state=42)

# %%
# Trigger SKD011 and SKD012
# =========================
#
# Let us train a gradient boosting model, using skrub's `TableVectorizer` to vectorize
# the data first.

from skore import evaluate

model = make_pipeline(vectorizer, regressor)

first_report = evaluate(
    model,
    X=X,
    y=y,
    splitter=splitter,
)
first_report

# %%
# We can see from the metrics table that the models performs very well on the data,
# with an R² of 0.96.
# However, looking at the checks, we can see in the Tips tab that SKD011 triggers on
# the `Average_Medicare_Payments` column.

first_report.checks.summarize()

# %%
# We should then question whether this column would be available in a real deployment.
# If it would, then we have found a good proxy for the target and we can move on.
# If it would not, then we should not use this column to train our model.
#
# In our case, the target (`Average_Total_Payments`) and the golden feature
# (`Average_Medicare_Payments`) come from the same process (billing aggregates).
# Therefore, `Average_Medicare_Payments` would not be available in a real deployment,
# and we should not use it to train our model. As a matter of fact,
# `Average_Covered_Charges` would also not be able in a deployment setting so we will
# also drop it.
#
# A final thing to be careful about is that SKD012 is triggered and tips us about
# `ProviderCity` and `Total_Discharges` being useless. As we can see in the importance
# plot below, every feature receives very low importance in the presence of the golden
# feature, when they may be signal rich and important in the absence of the golden
# feature. We will keep them for now.

_ = first_report.inspection.permutation_importance().plot()


# SKD011 - Inspect after dropping golden feature
# ==============================================
#
# First let's evaluate the same model on the same test set with the payment features
# removed, and compare it with our original model.
#
# Thankfully there is still signal in the remaining features, as we get a decent R²
# of 0.90.

from skore import compare

X_without_payment = X.drop(
    columns=["Average_Medicare_Payments", "Average_Covered_Charges"]
)

second_report = evaluate(
    model,
    X=X_without_payment,
    y=y,
    splitter=splitter,
)

comparison = compare(
    {
        "with_payment_features": first_report,
        "without_payment_features": second_report,
    }
)
comparison.metrics.summarize().frame()

# %%
# Let us inspect which features are now the ones our model relies on the most.
# We can see that `DRG_Definition` is by far most important one, according to
# permutation importance. This feature encodes the medical reason for which patients
# were treated. It will therefore be available at inference time and we can use it.

_ = second_report.inspection.permutation_importance().plot()

# %%
# SKD012 still flags `Total_Discharges`. This time it is not an artefact of a
# golden feature: the column stays weak without the payment aggregates. Let us
# drop it, and also the extra columns that `TableVectorizer` builds from
# high-cardinality strings.

second_report.checks.summarize()

# %%
# SKD012 - prune weak features inside the pipeline
# ================================================
#
# SKD012 only sees the original input columns. `TableVectorizer` turns
# high-cardinality fields such as `DRG_Definition` into many numeric
# components; some of those can be as uninformative as `Total_Discharges`.
# We therefore select *after* vectorizing, with
# :class:`~sklearn.feature_selection.SelectFromModel`, so the choice is fit on
# the training fold only.
#
# Histogram gradient boosting has no `feature_importances_`, so the selector
# uses :class:`~sklearn.ensemble.GradientBoostingRegressor`. The final
# estimator stays a :class:`~sklearn.ensemble.HistGradientBoostingRegressor` on
# the reduced matrix.

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel

model_reduced = make_pipeline(
    vectorizer,
    SelectFromModel(
        GradientBoostingRegressor(random_state=42),
        threshold="median",
    ),
    regressor,
)
model_reduced

# %%
third_report = evaluate(
    model_reduced,
    X=X_without_payment,
    y=y,
    splitter=splitter,
)

comparison_reduced = compare(
    {
        "without_payment_features": second_report,
        "without_payment_features_reduced": third_report,
    }
)
comparison_reduced.metrics.summarize().frame()

# %%
third_report.checks.summarize(fast_mode=True)

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
