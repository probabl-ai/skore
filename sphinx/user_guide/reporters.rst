.. _reporters:

================================
Evaluate your predictive models
================================

.. currentmodule:: skore

When you experiment with predictive models, a large part of the work is not novel:
you repeatedly evaluate performance, inspect how the model uses the data, and compare
candidates. That work still takes a lot of boilerplate and scattered code if you
assemble it by hand each time.

`skore` focuses that workflow behind :func:`~skore.evaluate` and **reports** that expose
a small, consistent API. A report is a structured object that answers questions about
**the data** used for evaluation, **how well** the model performs, and **what happens
inside** the model, without you wiring plots and tables from scratch.

Starting point: :func:`~skore.evaluate`
---------------------------------------

:func:`~skore.evaluate` is the main entry point to evaluate **one or several**
scikit-learn–compatible estimators on feature matrix ``X`` and target ``y``.

The ``splitter`` argument controls how ``X`` and ``y`` are split for evaluation. For a
**single train–test split**, pass a **float** (the default is ``0.2``) or a
:class:`~skore.TrainTestSplit` instance (see :func:`~skore.train_test_split`).
**Otherwise**, use any **scikit-learn cross-validation splitter** that follows the usual
scikit-learn cross-validation API.

Sometimes you only need to **evaluate a model that is already fitted**, without
refitting it. Pass ``splitter="prefit"`` in that case: ``X`` and ``y`` are treated as
the test set, and the estimator is not fit again.

For the full parameter list and return type rules, see the API documentation for
:func:`~skore.evaluate`.

Three report types, one layout
------------------------------

Depending on ``splitter`` and whether you pass one estimator or a list,
:func:`~skore.evaluate` returns one of:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Return type
     - Typical use
   * - :class:`EstimatorReport`
     - Single train–test split or prefitted estimator.
   * - :class:`CrossValidationReport`
     - Cross-validation with more than one split.
   * - :class:`ComparisonReport`
     - List of estimators evaluated together.

All three share the same **accessor** idea where it applies:

- ``.data`` — dataset-oriented analysis (:class:`EstimatorReport`,
  :class:`CrossValidationReport` only).
- ``.metrics`` — performance numbers, summaries, and task-specific diagnostics.
- ``.inspection`` — interpretability and internal structure of the model.

:class:`ComparisonReport` has **no** ``data`` accessor, because compared models may use
different inputs; use each underlying report in :attr:`~skore.ComparisonReport.reports_`
if you need per-model data views.

If you already have separate reports (same test target), you can also build a
:class:`ComparisonReport` with :func:`~skore.compare`.

What the accessors are for
--------------------------

Data insights (``report.data``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **data** accessor summarizes the dataset tied to the evaluation: useful views of
distributions, relationships between features, column types, missing values, and
similar exploratory signals. On a single train–test report you can often focus on
train, test, or both; on a cross-validation report, the analysis refers to the full
``X`` and ``y`` passed to :func:`~skore.evaluate`.

See :ref:`estimator_data` for the detailed API on :class:`EstimatorReport` (the
cross-validation accessor follows the same role on the full dataset).

Model evaluation (``report.metrics``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **metrics** accessor covers **statistical** performance (how good the predictions
are) and **computational** cost (how long fitting and predicting take). You can read
metrics **individually** or as a **summary table** via
:meth:`EstimatorReport.metrics.summarize` (and the corresponding accessors on
:class:`CrossValidationReport` and :class:`ComparisonReport`).

When you do not pass an explicit list of metrics, the defaults depend on the **machine
learning task** that `skore` infers from ``y`` and the estimator:

- **Classification**: accuracy, precision, and recall. For **binary** classification,
  ROC AUC is included as well, and if the estimator defines ``predict_proba``, Brier
  score is added. For **multiclass** classification, if ``predict_proba`` is
  available, ROC AUC and log loss are added to the defaults.
- **Regression**: R² (``r2``) and root mean squared error (``rmse``).

In all of these cases the default summary also includes **fit time** and **predict
time**. You can override the choice of metrics via the ``summarize`` API;
cross-validation and comparison reports support aggregating across folds or models where
relevant.

See :ref:`estimator_metrics` for :class:`EstimatorReport`,
:ref:`cross_validation_metrics` for :class:`CrossValidationReport`, and
:ref:`comparison_metrics` for :class:`ComparisonReport`.

Model interpretability (``report.inspection``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **inspection** accessor surfaces model internals and explanations that depend on
the estimator family: for example coefficients for linear models, or feature
importance–style summaries for tree-based models. What is available follows the
estimator you passed to :func:`~skore.evaluate`.

Technical notes
---------------

For a **walkthrough of the unified API** (the three report types, ``data`` /
``metrics`` / ``inspection``, and displays), see :ref:`example_skore_api`.

To use **scikit-learn–compatible estimators** from other libraries (e.g. gradient
boosting packages) or custom classes with a familiar ``fit`` / ``predict`` interface,
see :ref:`example_sklearn_api`. For ROC-based metrics, the estimator typically needs
``predict_proba`` when applicable, as noted in that example.
