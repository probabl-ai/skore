"""
.. _example_skore_api:

=============
The skore API
=============

This example illustrates the consistent API shared by skore reports and displays.
Reports expose the same accessors (``data``, ``metrics``, ``inspection``), and
each method that produces a visualization returns a **Display** object. All
displays implement a common interface: ``plot()``, ``frame()``, ``set_style()``,
and ``help()``.
"""

# %%
# Reports share the same accessor structure
# =========================================
#
# :class:`~skore.EstimatorReport`, :class:`~skore.CrossValidationReport`, and
# :class:`~skore.ComparisonReport` all expose the same accessors where applicable:
#
# - **data**: dataset analysis
# - **metrics**: performance metrics and related displays (e.g. ROC, confusion matrix)
# - **inspection**: model inspection (e.g. coefficients, feature importance)
#
# The ``data`` accessor is only available on :class:`~skore.EstimatorReport` and
# :class:`~skore.CrossValidationReport` because when comparing models, the input data
# can be different and thus one can access the underlying reports to inspect the data.
#
# Calling a method on these accessors returns a **Display** object. The same
# pattern holds across report types, so once you know one, you know them all.

# %%
# Minimal setup: one report and one display
# =========================================
#
# We build a simple :class:`~skore.EstimatorReport` and use it to show how
# accessors return displays and how those displays behave.
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from skore import EstimatorReport, train_test_split
from skrub import tabular_pipeline

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
split_data = train_test_split(X=X, y=y, random_state=42, as_dict=True)
estimator = tabular_pipeline(LogisticRegression())

report = EstimatorReport(estimator, **split_data)

# %%
# Data accessor: ``report.data.analyze()`` returns a display
# ----------------------------------------------------------
#
# The **data** accessor provides dataset summaries. Its ``analyze()`` method
# returns a :class:`~skore._sklearn._plot.data.table_report.TableReportDisplay`.
data_display = report.data.analyze()
data_display.help()

# %%
# Every display implements the same API. You can:
#
# - **Plot** it (with optional backend and style):
data_display.plot(kind="dist", x="mean radius", y="mean texture")

# %%
# You can set the style of the plot via ``set_style()`` and then call ``plot()``:
data_display.set_style(scatterplot_kwargs={"color": "orange", "alpha": 1.0})
data_display.plot(kind="dist", x="mean radius", y="mean texture")

# %%
# - **Export** the underlying data as a DataFrame:
data_display.frame()

# %%
# Metrics accessor: same idea, same display API
# =============================================
#
# The **metrics** accessor exposes methods such as ``confusion_matrix()``,
# ``roc_curve()``, ``precision_recall()``, and ``prediction_error()``. Each
# returns a display (e.g. :class:`~skore.ConfusionMatrixDisplay`) with the
# same interface: ``plot()``, ``frame()``, ``set_style()``, ``help()``.
metrics_display = report.metrics.confusion_matrix()
metrics_display.help()

# %%
metrics_display.frame()

# %%
# Draw the confusion matrix by calling ``plot()``:
metrics_display.plot()

# %%
# Inspection accessor
# ===================
#
# The **inspection** accessor exposes model-specific displays (e.g.
# ``coefficients()`` for linear models, ``impurity_decrease()`` for trees).
# These also return Display objects with the same ``plot()``, ``frame()``,
# ``set_style()``, and ``help()`` methods.
inspection_display = report.inspection.coefficients()
inspection_display.plot(select_k=15, sorting_order="descending")

# %%
# Same API with :class:`~skore.CrossValidationReport`
# ===================================================
#
# The same accessors and display API apply to :class:`~skore.CrossValidationReport`.
# We use the same dataset and model; only the report type changes.
from skore import CrossValidationReport

cv_report = CrossValidationReport(estimator, X, y, splitter=3)

# %%
# Again: ``data``, ``metrics``, and ``inspection`` return displays with
# ``plot()``, ``frame()``, and ``set_style()``.
cv_report.data.analyze().plot(kind="dist", x="mean radius", y="mean texture")

# %%
cv_report.metrics.confusion_matrix().plot()

# %%
cv_report.inspection.coefficients().plot(select_k=10, sorting_order="descending")

# %%
# The same accessors and display API apply to :class:`~skore.ComparisonReport`
# (metrics and inspection; no data accessor when comparing reports).

# %%
# Summary
# =======
#
# - **Reports** (Estimator, CrossValidation, Comparison) use the same accessor
#   layout: ``report.data``, ``report.metrics``, ``report.inspection`` (where
#   applicable).
# - **Accessor methods** that produce figures or tables return **Display**
#   objects.
# - **Displays** share a single, predictable API:
#
#   - ``plot(**kwargs)`` — render the visualization
#   - ``frame(**kwargs)`` — return the data as a :class:`pandas.DataFrame`
#   - ``set_style(policy=..., **kwargs)`` — customize appearance
#   - ``help()`` — show available options
#
# This consistency makes it easy to switch between report types and to reuse
# the same workflow across data, metrics, and inspection.
