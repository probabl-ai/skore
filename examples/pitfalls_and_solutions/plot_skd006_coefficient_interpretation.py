"""
.. _example_skd006_coefficient_interpretation:

SKD006 - Coefficient interpretation
===================================

:ref:`SKD006 <skd006-unscaled-coefficients>` is a tip about interpretation.
This check flags that on mixed-scale features, raw coefficient magnitudes are
not directly comparable across columns, and after standardization they are
comparable but no longer expressed in the original feature units. Which path
you take depends on what you want to read from the coefficients. See also
scikit-learn's example on `linear model coefficient interpretation
<https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html>`_.

When reading coefficients under SKD006, think of:

- standardizing the inputs when you want coefficients that are comparable
  across features (see `scale matters
  <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#interpreting-coefficients-scale-matters>`_),
- multiplying each coefficient by the feature's standard deviation for an
  "effect per one std" without refitting,
- relying on a scale-invariant ranking such as permutation importance when you
  need importance that does not depend on coefficient units.

We use California housing regression with
:class:`~sklearn.linear_model.Ridge` on raw numeric columns. The goal is to
interpret effect sizes without mistaking scale differences for importance.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Median income is measured in 10k USD blocks, latitude in degrees, and
# population in head counts. Fitting Ridge on these raw columns produces
# coefficients whose magnitudes reflect units as much as predictive strength.
# We load the table via skrub so Sphinx CI does not depend on scikit-learn's
# ``cal_housing.tgz`` download cache.

from skrub.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.X, housing.y

# %%
# :class:`~skrub.TableReport` highlights the range differences across columns.

from skrub import TableReport

TableReport(X)

# %%

TableReport(y)

# %%
# A single :class:`~skore.TrainTestSplit` with a 20 % test set feeds every
# evaluation below.

from skore import TrainTestSplit

splitter = TrainTestSplit(test_size=0.2, random_state=42)

# %%
# Trigger SKD006 with Ridge on raw mixed-scale features
# =====================================================
#
# Ridge with a moderate ``alpha`` fits quickly on unscaled inputs. SKD006
# indicates that the features are not on the same scale, so raw coefficient
# magnitudes should not be read as a ranking of importance.

from sklearn.linear_model import Ridge
from skore import evaluate

report = evaluate(
    Ridge(alpha=1.0),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# Find ``SKD006`` in the Tips tab below.

report.checks.summarize()

# %%
# Inspect coefficients with
# :meth:`~skore.EstimatorReport.inspection.coefficients`.
# Compare each coefficient's magnitude to the feature's typical range (or
# standard deviation): a large coefficient on a small-scale column is not
# necessarily more important than a small coefficient on a large-scale column
# such as ``Population``.
#
# Unscaled coefficients are still useful on their own: they answer "if I change
# this feature by one unit in its original scale, how much does the prediction
# change in target units?" They are misleading regarding feature importance.
coef_display = report.inspection.coefficients()
coef_display.frame()

# %%
# Side by side, the raw coefficient magnitudes and the feature standard
# deviations tell different stories. ``AveBedrms`` often carries one of the
# largest absolute coefficients while having a small standard deviation, so its
# "per unit bedroom" effect looks large. ``Population`` has a tiny coefficient
# because a one-person change is negligible on a head-count scale, even though
# the column varies a lot across districts.

import matplotlib.pyplot as plt

coef = coef_display.frame()
coef_no_intercept = coef[coef["feature"] != "Intercept"].set_index("feature")
feature_std = report.X_train.std()

_, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
_ = coef_no_intercept["coefficient"].plot.barh(ax=axes[0], legend=False)
axes[0].set_title("Raw Ridge coefficients")
axes[0].set_xlabel("coefficient")
_ = feature_std.loc[coef_no_intercept.index].plot.barh(
    ax=axes[1], legend=False, color="C1"
)
axes[1].set_title("Feature standard deviation (train)")
axes[1].set_xlabel("std")

# %%
# Scale coefficients by feature standard deviation
# ================================================
#
# Recall the pitfall above: when the model was trained on features with
# different dynamic ranges, you cannot compare features by looking at raw
# coefficients alone. Multiplying each fitted coefficient by its feature
# standard deviation gives an "effect per one standard deviation" that is
# comparable across columns without refitting. See scikit-learn's discussion of
# `scale matters
# <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#interpreting-coefficients-scale-matters>`_.

comparable = coef_no_intercept.copy()
comparable["effect_per_std"] = comparable["coefficient"] * feature_std
comparable.sort_values("effect_per_std", key=abs, ascending=False)

# %%
# After this rescaling, ``AveBedrms`` usually drops in the ranking: its large
# raw coefficient was inflated by the small bedroom-count scale. Features such
# as ``MedInc``, ``Latitude``, or ``Longitude`` move up when importance is
# measured per one standard deviation rather than per original unit.

# %%
# Standardize inputs in a pipeline
# ================================
#
# :class:`~sklearn.preprocessing.StandardScaler` inside a
# :class:`~sklearn.pipeline.Pipeline` puts every feature on a common scale at
# fit time, so coefficients become directly comparable as "effect per one
# standard deviation." That is useful for ranking features, but you lose
# statements in original units (for example, "one extra year of age"). SKD006
# then tips that the coefficients are comparable but no longer in the original
# feature units: the other side of the same interpretation trade-off. See again
# `scale matters
# <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#interpreting-coefficients-scale-matters>`_.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

report_scaled = evaluate(
    Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
# ``SKD006`` is still reported in the Tips tab but the warning message changed.

report_scaled.checks.summarize()

# %%
# The coefficient table is now on a shared scale. Read magnitudes as relative
# importance after standardization, not as effects per original unit of income,
# rooms, or population.
#
# Up to small numerical differences (train-fold vs. scaler internals), these
# values match the post-hoc ``coefficient * feature_std`` column from the
# previous section: both express an effect per one standard deviation.

report_scaled.inspection.coefficients().frame()

# %%
# Permutation importance (scale-invariant)
# ========================================
#
# Scaled coefficients are one way to rank features. Another is permutation
# importance on held-out data: shuffle one column at a time and measure how
# much the score drops. That answers questions such as "which features did the
# model rely on most?" and works for nonlinear models as well. Present it as an
# alternative to scaled coefficients when you care about predictive reliance
# rather than a linear effect size.
#
# Correlated features can distort both coefficient rankings and permutation
# importance (credit is shared or shifted between partners). See
# :ref:`SKD008 <skd008-correlated-features>` and scikit-learn's note on
# `misleading values on strongly correlated features
# <https://scikit-learn.org/stable/modules/permutation_importance.html#misleading-values-on-strongly-correlated-features>`_.

display = report.inspection.permutation_importance(
    seed=42,
    n_repeats=5,
)

# %%
display.frame()

# %%
display.plot()

# %%
# Conclusion
# ==========
#
# SKD006 is a reminder to be careful when interpreting coefficients of linear
# models. When features are on different scales, linear coefficients mix
# feature importance with feature magnitude; when features are standardized,
# coefficients are comparable but no longer in original units.
#
# Unscaled coefficients remain useful when the question is "if I change this
# feature by this much in its own units, how much does the prediction change?"
# Choose standardization, an effect-per-std rescaling, or a scale-invariant
# method such as permutation importance depending on the question you want to
# answer.
