"""
.. _example_skd010_slower_than_baseline:

SKD010 - Model slower than baseline
===================================

This example walks through mitigations when the check
:ref:`SKD010 <skd010-slower-than-baseline>` fires. The check compares the user's
model to a fast linear baseline
(:class:`~sklearn.linear_model.RidgeCV` for regression, wrapped in
:func:`~skrub.tabular_pipeline`) and flags a problem only when both of the
following hold:

- the larger of the fit-time and test predict-time ratios is at least 2 times
  the baseline, with an absolute gap of at least 1 second on that dimension,
  and
- test scores are not significantly better than the baseline on a majority of
  default predictive metrics.

The issue is paying for latency without a significant quality premium. A clear
quality gain over RidgeCV keeps the check quiet. So does a model that is
already about as fast as the baseline.

Responses tried below, in order:

- check that preprocessing is not the actual bottleneck,
- reduce the model's complexity,
- switch to the fast linear baseline when quality is sufficient,
- profile fit time to understand the dominant cost.

We use the employee salaries dataset with a heavy random forest inside
:func:`~skrub.tabular_pipeline`. The goal is either to match RidgeCV quality at
lower cost, or to shrink fit time until the speed gap is no longer unjustified.
"""

# %%
# Load the employee salaries dataset
# ==================================
#
# :func:`skrub.datasets.fetch_employee_salaries` returns human-resources records
# with mixed categorical and numeric fields. A 200-tree random forest inside
# ``tabular_pipeline`` trains slowly yet often fails to beat the fast RidgeCV
# baseline on test scores. SKD010 is a slow check.

from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
X = dataset.X
y = dataset.y

# %%
# Inspect column types with :class:`~skrub.TableReport`. Encoding cost is part
# of total fit time.

from skrub import TableReport

TableReport(X)

# %%
# Salaries are continuous and right-skewed, a typical regression target.

TableReport(y)

# %%
from skore import TrainTestSplit

splitter = TrainTestSplit(test_size=0.2, random_state=42)

# %%
# Trigger SKD010 with large leaves
# ================================
#
# One hundred trees that must keep 100 training rows in every leaf barely
# split. :func:`~skrub.tabular_pipeline` still encodes the high-cardinality
# strings, so the fit stays much slower than RidgeCV, and the test scores lose
# the quality premium a forest can have on this table.

from sklearn.ensemble import RandomForestRegressor
from skore import evaluate
from skrub import tabular_pipeline

report_constrained = evaluate(
    tabular_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=100,
            random_state=42,
            n_jobs=4,
        )
    ),
    X=X,
    y=y,
    splitter=splitter,
)
report_constrained

# %%
# ``SKD010`` is in the issues below: fit time is several times the RidgeCV
# baseline, and the test scores are not significantly better.

report_constrained.checks.summarize()

# %%
# Fit the fast linear baseline on the same split. The next two tables are the
# two gates: fit time and test predict time, then the predictive scores.
from sklearn.linear_model import RidgeCV

report_linear = evaluate(
    tabular_pipeline(RidgeCV()),
    X=X,
    y=y,
    splitter=splitter,
)
report_linear

# %%
from skore import compare

compare({"large_leaves": report_constrained, "ridge": report_linear}).metrics.summarize(
    data_source="test",
).frame()

# %%
# Fit time clears the 2x / 1s gate. R², RMSE, and MAE stay close to RidgeCV,
# short of ``max(0.01, 0.05 * |baseline|)`` on a majority of the default
# predictive metrics. That is the warning.

# %%
# Grow the trees to full depth
# ============================
#
# Drop the leaf limit and keep the same 100 trees. The fit gets slower. The
# test scores move far enough ahead of RidgeCV that SKD010 passes.

report_full = evaluate(
    tabular_pipeline(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
    ),
    X=X,
    y=y,
    splitter=splitter,
)
report_full

# %%
# ``SKD010`` is among the passed checks. Slowness alone is not the warning.

report_full.checks.summarize()

# %%
compare(
    {
        "large_leaves": report_constrained,
        "full_depth": report_full,
        "ridge": report_linear,
    }
).metrics.summarize(
    metric=["fit_time", "predict_time"],
    data_source="test",
).frame()

# %%
compare(
    {
        "large_leaves": report_constrained,
        "full_depth": report_full,
        "ridge": report_linear,
    }
).metrics.summarize(data_source="test").frame()

# %%
# Fit time increased from the constrained forest, and R² cleared the quality
# bar. The warning is gone.

# %%
# Use a cheaper encoder
# =====================
#
# Sometimes you can spend less time before touching the forest. Drop a column you do not
# need: ``date_first_hired`` repeats ``year_first_hired``, and a string encoder still
# has to run on its thousands of distinct dates. Or keep the columns and pick a cheaper
# encoder. A tree can split on category codes, so an
# :class:`~sklearn.preprocessing.OrdinalEncoder` is enough for the high-cardinality
# strings that :func:`~skrub.tabular_pipeline` otherwise sends to
# :class:`~skrub.StringEncoder`.

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from skrub import TableVectorizer

report_ordinal = evaluate(
    make_pipeline(
        TableVectorizer(
            high_cardinality=OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
        ),
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4),
    ),
    X=X,
    y=y,
    splitter=splitter,
)
report_ordinal

# %%
compare(
    {"full_depth": report_full, "ordinal_encoder": report_ordinal}
).metrics.summarize(
    metric=["fit_time", "predict_time"],
    data_source="test",
).frame()

# %%
# The ordinal encoder cuts fit time and predict time. Test R² shows what that
# speed costs relative to the string encoder.

compare(
    {"full_depth": report_full, "ordinal_encoder": report_ordinal}
).metrics.summarize(metric=["r2"], data_source="test").frame()

# %%
# Reduce model complexity
# =======================
#
# Keep the 100 trees and the full depth, and stop splits that would leave a
# leaf with fewer than 20 training rows. ``min_samples_leaf`` shortens each
# tree, which cuts fit time and predict time on the string-encoder pipeline.

report_leaf = evaluate(
    tabular_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=4,
        )
    ),
    X=X,
    y=y,
    splitter=splitter,
)
report_leaf

# %%
# SKD010 stays passed. The larger leaves shorten fit time on the same 100 trees.

report_leaf.checks.summarize()

# %%
report_leaf.metrics.summarize(
    metric=["fit_time", "predict_time"],
    data_source="test",
).frame()

# %%
# Compare fit and predict times
# =============================
#
# Fit time and test predict time for every pipeline above. Large leaves are the
# case that raises SKD010. Full depth is slower and justified. The ordinal
# encoder and a moderate leaf limit spend less time. RidgeCV is the speed
# reference.

compare(
    {
        "large_leaves": report_constrained,
        "full_depth": report_full,
        "ordinal_encoder": report_ordinal,
        "moderate_leaves": report_leaf,
        "ridge": report_linear,
    }
).metrics.summarize(
    metric=["fit_time", "predict_time"],
    data_source="test",
).frame()

# %%
# Compare predictive metrics
# ==========================
#
# Test scores next to the timings above. SKD010 asks whether extra seconds buy
# a significant score gain over RidgeCV.

compare(
    {
        "large_leaves": report_constrained,
        "full_depth": report_full,
        "ordinal_encoder": report_ordinal,
        "moderate_leaves": report_leaf,
        "ridge": report_linear,
    }
).metrics.summarize(
    metric=["r2", "rmse", "mae"],
    data_source="test",
).frame()

# %%
# Conclusion
# ==========
#
# A usual forest can be slower than RidgeCV and still pass SKD010 when test
# scores are significantly better. A cheaper high-cardinality encoder, or
# dropping a redundant column such as ``date_first_hired``, cuts that cost.
# ``min_samples_leaf`` shrinks the trees while keeping the same number of
# trees. RidgeCV remains the speed reference when a linear model is enough.
