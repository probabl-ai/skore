"""
.. _example_skd006_coefficient_interpretation:

SKD006 — Coefficient interpretation
===================================

This example walks through mitigations when check
:ref:`SKD006 <skd006-unscaled-coefficients>` tips on a linear model fitted on
mixed-scale features. The check compares per-feature standard deviations and
warns when raw coefficient magnitudes are not directly comparable — or when
scaled coefficients are comparable but no longer in original units.

Mitigations from the :ref:`automated_checks` user guide:

- standardize the inputs to make coefficients comparable,
- multiply each coefficient by the feature's standard deviation,
- rely on scale-invariant importance such as permutation importance.

We use California housing regression with :class:`~sklearn.linear_model.Ridge`
on raw numeric columns. The goal is to interpret effect sizes without mistaking
scale differences for importance.
"""

# %%
# Load the California housing dataset
# ===================================
#
# Median income is measured in \$10k blocks, latitude in degrees, and
# population in head counts. Fitting Ridge on these raw columns produces
# coefficients whose magnitudes reflect units as much as predictive strength.

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
# A single :class:`~skore.TrainTestSplit` feeds every evaluation below.

from skore import TrainTestSplit

splitter = TrainTestSplit(random_state=42)

# %%
# Trigger SKD006 — Ridge on raw mixed-scale features
# ==================================================
#
# Ridge with a moderate ``alpha`` fits quickly on unscaled inputs. Inspect
# coefficients through :meth:`~skore.EstimatorReport.inspection.coefficients`,
# then run checks — SKD006 should note that features are not on the same scale.

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
report.inspection.coefficients().frame()

# %%
# Scale coefficients by feature standard deviation
# ================================================
#
# Multiplying each fitted coefficient by its feature standard deviation gives
# an "effect per one standard deviation" that is comparable across columns
# without refitting the model.

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
# :class:`~sklearn.preprocessing.StandardScaler` inside a
# :class:`~sklearn.pipeline.Pipeline` puts coefficients on a common scale at
# fit time. SKD006 then notes that the coefficients are comparable but no longer
# in original units;
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
report_scaled.inspection.coefficients().frame()

# %%
# Permutation importance (scale-invariant)
# ========================================
#
# When features need to be interpreted in original units, or when the
# model is nonlinear, permutation importance on held-out data offers a
# scale-invariant ranking that does not depend on coefficient magnitudes.

import matplotlib.pyplot as plt

perm_display = report.inspection.permutation_importance(
    seed=42,
    n_repeats=5,
)
perm_display.frame()

# %%
fig = perm_display.plot()
plt.show()

# %%
# Conclusion
# ==========
#
# SKD006 is a reminder on whether the features are on the same scale or not and
# what each case entails. When the features are not on the same scale, on a
# linear model the coefficiants are influence by the magnitude of the feature
# as well as its importance, and when the features are on the same scale, the
# coefficients are not intrepretable in their original units.
