"""
.. _example_skd006_coefficient_interpretation:

SKD006 — Coefficient interpretation
===================================

:ref:`SKD006 <skd006-unscaled-coefficients>` is a tip about interpretation, not a sign that the model failed. This check
only flags that on mixed-scale features, raw coefficient magnitudes are not directly comparable across columns, and
after standardization they are comparable but no longer expressed in the original feature units. Which path you take
depends on what you want to read from the coefficients. See also scikit-learn's example on `linear model coefficient interpretation <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html>`_.

In this case, think of:

- standardizing the inputs when you want coefficients that are comparable across features (see `scale matters <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#interpreting-coefficients-scale-matters>`_),
- multiplying each coefficient by the feature's standard deviation for an "effect per one std" without refitting (same
  sklearn discussion),
- relying on a scale-invariant ranking such as permutation importance when you need importance that does not depend on
  coefficient units.

We use California housing regression with :class:`~sklearn.linear_model.Ridge` on raw numeric columns. The goal is to
interpret effect sizes without mistaking scale differences for importance.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Median income is measured in \$10k blocks, latitude in degrees, and population in head counts. Fitting Ridge on these
# raw columns produces coefficients whose magnitudes reflect units as much as predictive strength.

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X = housing.frame.drop(columns=["MedHouseVal"])
y = housing.frame["MedHouseVal"]

# %%
# :class:`~skrub.TableReport` highlights the range differences across columns.

from skrub import TableReport

TableReport(X)

# %%
# The target is continuous house value in \$100k units.

TableReport(y)

# %%
# A single :class:`~skore.TrainTestSplit` with a 20 % test set feeds every evaluation below.

from skore import TrainTestSplit

splitter = TrainTestSplit(test_size=0.2, random_state=42)

# %%
# Trigger SKD006 — Ridge on raw mixed-scale features
# ==================================================
#
# Ridge with a moderate ``alpha`` fits quickly on unscaled inputs. SKD006 is not a sign that the fit failed — other
# issues may still exist — but it does flag that the features are not on the same scale, so raw coefficient magnitudes
# should not be read as a ranking of importance.

from sklearn.linear_model import Ridge
from skore import evaluate

report = evaluate(
    Ridge(alpha=1.0),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
report.checks.summarize()

# %%
# Inspect coefficients with :meth:`~skore.EstimatorReport.inspection.coefficients`. Compare each coefficient's magnitude
# to the feature's typical range (or standard deviation): a large coefficient on a small-scale column is not necessarily
# more important than a small coefficient on a large-scale column such as population.

report.inspection.coefficients().frame()

# %%
# Scale coefficients by feature standard deviation
# ================================================
#
# Recall the pitfall above: when the model was trained on features with different dynamic ranges, you cannot compare
# features by looking at raw coefficients alone. Multiplying each fitted coefficient by its feature standard deviation
# gives an "effect per one standard deviation" that is comparable across columns without refitting. See scikit-learn's
# discussion of `scale matters <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#interpreting-coefficients-scale-matters>`_.

coef = report.inspection.coefficients().frame()
feature_std = X.std()

comparable = coef[coef["feature"] != "Intercept"].copy()
comparable["effect_per_std"] = comparable["coefficient"] * comparable["feature"].map(
    feature_std
)
comparable.sort_values("effect_per_std", key=abs, ascending=False)

# %%
# Standardize inputs in a pipeline
# ================================
#
# :class:`~sklearn.preprocessing.StandardScaler` inside a :class:`~sklearn.pipeline.Pipeline` puts every feature on a
# common scale at fit time, so coefficients become directly comparable as "effect per one standard deviation." That is
# useful for ranking features, but you lose statements in original units (for example, "one extra year of age"). SKD006
# then tips that the coefficients are comparable but no longer in the original feature units — the other side of the
# same interpretation trade-off. See again `scale matters <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#interpreting-coefficients-scale-matters>`_.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

report_scaled = evaluate(
    Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
    X=X,
    y=y,
    splitter=splitter,
)

# %%
report_scaled.checks.summarize()

# %%
# The coefficient table is now on a shared scale. Read magnitudes as relative importance after standardization, not as
# effects per original unit of income, rooms, or population.

report_scaled.inspection.coefficients().frame()

# %%
# Permutation importance (scale-invariant)
# ========================================
#
# When you need a ranking that does not depend on coefficient units — for example because you want to keep features in
# original units, or the model is nonlinear — permutation importance on held-out data is a scale-invariant alternative.

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
# SKD006 is a reminder about scale and interpretation, not a sign that the fit failed (other problems may still be
# present). When features are on different scales, linear coefficients mix feature importance with feature magnitude;
# when features are standardized, coefficients are comparable but no longer in original units. Choose standardization,
# an effect-per-std rescaling, or a scale-invariant method such as permutation importance depending on the question you
# want to answer.
