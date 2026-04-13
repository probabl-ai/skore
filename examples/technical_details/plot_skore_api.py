"""
.. _example_skore_api:

=============
The skore API
=============

Skore has three types of reports: :class:`~skore.EstimatorReport`
(single train-test evaluation), :class:`~skore.CrossValidationReport`
(cross-validation), and :class:`~skore.ComparisonReport` (comparing several
estimators). All three are created via :func:`~skore.evaluate` by passing an
estimator (or a list or dict of named estimators for comparison), the data ``X``
and ``y``, and a ``splitter`` that controls the evaluation strategy.

This example showcases the **unified API** shared by these reports: they expose
the same accessors (``data``, ``metrics``, ``inspection``). Methods that
produce a visualization return a **Display** object with ``plot()``, ``frame()``,
``set_style()``, and ``help()``.
"""

# %%
# Three report types, one API
# ===========================
#
# :func:`~skore.evaluate` returns one of three report types depending on its
# ``splitter`` argument: an :class:`~skore.EstimatorReport` when ``splitter`` is a
# float or ``"prefit"``, a :class:`~skore.CrossValidationReport` when ``splitter`` is
# an integer or a scikit-learn cross-validator (e.g. ``KFold``, ``StratifiedKFold``),
# or a :class:`~skore.ComparisonReport` when passing a list or dict of estimators.
# All three respect the same accessor layout where applicable:
#
# - **data**: dataset analysis
# - **metrics**: performance metrics and related displays (e.g. ROC, confusion matrix)
# - **inspection**: model inspection (e.g. coefficients, feature importance)
#
# The ``data`` accessor is not available on ComparisonReport because compared
# models may use different input data; you can still inspect each underlying report.
# Methods on these accessors return **Display** objects with a common interface.

# %%
# First report: single train-test split
# =====================================
#
# We call :func:`~skore.evaluate` with the default ``splitter=0.2`` to get an
# :class:`~skore.EstimatorReport`. The accessors and display API shown below
# are the same for the other report types.
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from skore import evaluate
from skrub import tabular_pipeline

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
estimator = tabular_pipeline(LogisticRegression())

report = evaluate(estimator, X, y, splitter=0.2)

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
# Second report type: cross-validation
# ====================================
#
# Using the same :func:`~skore.evaluate` with an integer ``splitter`` returns a
# :class:`~skore.CrossValidationReport`. The same accessors and display API
# apply; only the way the report was built changes.
cv_report = evaluate(estimator, X, y, splitter=3)

# %%
# Again: ``data``, ``metrics``, and ``inspection`` return displays with
# ``plot()``, ``frame()``, and ``set_style()``.
cv_report.data.analyze().plot(kind="dist", x="mean radius", y="mean texture")

# %%
cv_report.metrics.confusion_matrix().plot()

# %%
cv_report.inspection.coefficients().plot(select_k=10, sorting_order="descending")

# %%
# Third report type: comparison
# =============================
#
# Passing a **list or dict of estimators** to :func:`~skore.evaluate` returns a
# :class:`~skore.ComparisonReport`. It exposes the same ``metrics`` and
# ``inspection`` accessors (no ``data`` accessor, since compared models can
# use different datasets). The display API is unchanged.

# %%
# Summary
# =======
#
# - **Three report types** (:class:`~skore.EstimatorReport`,
#   :class:`~skore.CrossValidationReport`, :class:`~skore.ComparisonReport`) are
#   all created with :func:`~skore.evaluate` and share the same accessor layout:
#   ``report.data``, ``report.metrics``, ``report.inspection`` (where applicable).
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
