.. Template for a new "Unreleased" section (hidden from docs).
   When releasing, copy-paste at the top of the changelog, and remove the indentation.

   `Unreleased`_
   =============

   .. _Unreleased: https://github.com/probabl-ai/skore/compare/skore/<new-version>...HEAD

   Release highlights
   ------------------

   Changed
   -------

   Added
   -----

   Removed
   -------

   Fixed
   -----

.. The changelog follows conventions close to https://common-changelog.org

.. _changes:

=========
Changelog
=========

.. currentmodule:: skore

`Unreleased`_
=============

.. _Unreleased: https://github.com/probabl-ai/skore/compare/skore/0.15.0...HEAD

Release highlights
------------------

Changed
-------

- **Breaking change:** ``cache_predictions`` no longer accepts ``n_jobs`` or
  ``response_methods`` on estimator, cross-validation, and comparison reports.
  It now infers and caches the relevant prediction outputs automatically. See
  :pr:`2677` by :user:`cakedev0`.

- **Breaking change:** Computing custom metrics on reports is now done by calling
  :meth:`EstimatorReport.metrics.add` followed by :meth:`summarize`.
  :meth:`custom_metric` has been removed, and :meth:`summarize` no longer accepts
  callables or keyword arguments; only names of registered metrics are supported
  (which includes the default metrics such as `accuracy` for classifiers).
  Example usage is available `here <https://docs.skore.probabl.ai/dev/auto_examples/model_evaluation/plot_custom_metrics.html>`__.
  See :pr:`2736` by :user:`auguste-probabl`.

- Python `3.14.4` is now supported, while Python `3.10` support is now deprecated. See :pr:`2771` by :user:`thomass-dev`.

Added
-----

- :meth:`~EstimatorReport.metrics.summarize` (and equivalent methods on
  :class:`~skore.CrossValidationReport` and :class:`~skore.ComparisonReport`) now
  accept scikit-learn metric names without the `neg_` prefix. For example,
  ``"mean_squared_error"`` can be passed instead of ``"neg_mean_squared_error"``
  and is resolved automatically. See :pr:`2735` by :user:`direkkakkar319-ops`.
- Reports now expose a `report.diagnose()` method that automatically
  detects common modeling issues such as overfitting and underfitting.
- :class:`~skore.EstimatorReport` and :class:`~skore.CrossValidationReport` now
  expose :meth:`get_state` / :meth:`from_state` helpers to support more robust
  serialization and deserialization across skore versions than plain pickling.
  See :pr:`2741` by :user:`cakedev0`.
- :meth:`~EstimatorReport.metrics.mae`,
  :meth:`~EstimatorReport.metrics.map` (and equivalent methods on
  :class:`~skore.CrossValidationReport` and :class:`~skore.ComparisonReport`) are
  now available as default regression metrics, exposing mean absolute error (MAE)
  and mean absolute percentage error (MAP). See :pr:`2695` by :user:`direkkakkar319-ops`.

Removed
-------

Fixed
-----

`0.15.0`_ (2026-04-02)
======================

.. _0.15.0: https://github.com/probabl-ai/skore/compare/skore/0.14.0...skore/0.15.0

Changed
-------

- **Breaking change:** :meth:`Display.plot` (and all concrete displays) now returns a
  :class:`matplotlib.figure.Figure` instead of storing plotting artifacts on the display.
  The attributes ``figure_``, ``ax_``, and ``facet_`` are no longer set. Use
  ``fig = display.plot(...)`` and then ``fig.axes``, ``fig.show()``, or rely on the
  figure's rich representation in notebooks. :pr:`2660` by :user:`glemaitre`.

- **Breaking change:** :class:`~skore.EstimatorReport` now requires ``X_test``. If
  ``X_test`` is omitted (``None``), construction raises ``ValueError``. :pr:`2673` by
  :user:`glemaitre`.

- **Breaking change:** binary classification metrics and curve displays no longer
  require ``pos_label`` to be set, and skore no longer infers it implicitly. When
  ``pos_label`` is left unset, ``precision`` and ``recall`` metrics, as well as
  ROC and precision-recall curves, now expose both classes instead of failing.
  :pr:`2663` by :user:`cakedev0`.

Added
-----

- Add rich HTML representation for the different reports. See :pr:`2632`, :pr:`2651`,
  :pr:`2658`, and :pr:`2659` by :user:`glemaitre`, :user:`jeromedockes` and
  :user:`GaelVaroquaux`.


`0.14.0`_ (2026-03-19)
======================

.. _0.14.0: https://github.com/probabl-ai/skore/compare/skore/0.13.1...skore/0.14.0

Release highlights
------------------

- :func:`skore.evaluate` and :func:`skore.compare` are new top-level dispatcher functions that create the appropriate report class from an estimator in a single function call. See :pr:`2573` by :user:`raotalha71` and :pr:`2604` by :user:`glemaitre`.
- skore can now integrate with MLflow by passing `mode="mlflow"` and the new `tracking_uri` option to :class:`~skore.Project`. Example usage is available `here <https://docs.skore.probabl.ai/dev/auto_examples/technical_details/plot_skore_mlflow_project.html>`__. See :pr:`2527` by :user:`cakedev0`.

Changed
-------

- **Breaking:** The filtering and display arguments previously accepted by ``.summarize()``, e.g. `aggregate`, `flat_index`, `favorability`, have been moved to :meth:`~MetricsSummaryDisplay.frame()`, for every report. See :pr:`2536`, :pr:`2545`, and :pr:`2566` by :user:`auguste-probabl`.
- **Breaking:** :meth:`Display.set_style` now returns ``None`` instead of ``self``. Method chaining such as ``display.set_style(...).plot()`` is no longer supported. See :pr:`2579` by :user:`direkkakkar319-ops`.
- **Breaking:** ``pos_label`` can no longer be overridden when calling a metric or a display; it must be set when creating the report. See :pr:`2588` by :user:`jeromedockes`.
- **Breaking:** The ``name`` parameter of :meth:`ComparisonReport.create_estimator_report` has been renamed to ``report_key``. See :pr:`2561` by :user:`GaetandeCast`.

Added
-----

- :class:`PermutationImportanceDisplay` now supports a ``level`` parameter to select which level of a multi-index DataFrame to use for feature names. See :pr:`2565` by :user:`GaetandeCast`.
- :class:`CoefficientsDisplay` now supports aggregation via an ``aggregate`` parameter in :meth:`~CoefficientsDisplay.frame`, following the same pattern as :class:`ImpurityDecreaseDisplay`. See :pr:`2552` by :user:`MuditAtrey`.
- :class:`PermutationImportanceDisplay` and :class:`ImpurityDecreaseDisplay` now support parameters `select_k` and `sorting_order`, with the same behaviour already available in :class:`CoefficientsDisplay`. Passing `select_k=0` now raises a clearer error. See :pr:`2591` by :user:`GaetandeCast`.

Removed
-------

- **Breaking:** The ``data_source="X_y"`` option has been removed from report classes; ``"train"`` and ``"test"`` are now the only options. See :pr:`2537` by :user:`jeromedockes`.

Fixed
-----

- :class:`CrossValidationReport.metrics.summarize` now raises a ``NotImplementedError`` upon ``data_source="both"``. See :pr:`2548` by :user:`auguste-probabl`.
- Passing sparse matrices to data accessors now raises a ``NotImplementedError`` instead of crashing with an unhelpful error. See :pr:`2543` by :user:`KaranSinghDev`.

`0.13.1`_ (2026-03-05)
======================

.. _0.13.1: https://github.com/probabl-ai/skore/compare/skore/0.13.0...skore/0.13.1

Release highlights
------------------

- :class:`ComparisonReport` now supports permutation importance through :func:`~ComparisonReport.inspection.permutation_importance`. See :pr:`2511` by :user:`GaetandeCast`.
- :meth:`ImpurityDecreaseDisplay.frame` now supports aggregation through an `aggregate` parameter. See :pr:`2539` by :user:`MuditAtrey`.

Fixed
-----

- The figure-level legend is no longer cut off when saving the figure produced by `PredictionErrorDisplay.plot()`. See :pr:`2530` by :user:`MuditAtrey`.

`0.13.0`_ (2026-02-26)
======================

.. _0.13.0: https://github.com/probabl-ai/skore/compare/skore/0.12.0...skore/0.13.0

Release highlights
------------------

- Help menus accessed through ``help()`` methods now render as interactive HTML in Jupyter notebooks and IPython environments. See :pr:`2316` by :user:`glemaitre`.

Changed
-------

- **Breaking:** Reports no longer accept clustering models and only support supervised learning tasks (classification and regression). See :pr:`2489` by :user:`GaetandeCast`.
- **Breaking:** The ``mode`` parameter is now required when creating a :class:`Project` in "hub" mode. See :pr:`2401` by :user:`thomass-dev`.
- DataFrame column names in reports now consistently use singular form (e.g., "model" instead of "models"). See :pr:`2392` by :user:`Sharkyii`.
- Exceptions raised during the :class:`CrossValidationReport` fitting process are no longer caught by skore. See :pr:`2462` by :user:`glemaitre`.

Added
-----

- The documentation now includes a short example illustrating local usage of :class:`skore.Project`. See :pr:`2481` by :user:`glemaitre`.
- The documentation now includes a short example explaining the design philosophy of the skore API. See :pr:`2480` by :user:`glemaitre`.
- The report URL is now printed after uploading to Skore Hub. See :pr:`2488` by :user:`rouk1`.
- Reports now support setting ``pos_label`` after initialization, enabling users to change the positive class for binary classification metrics. See :pr:`2438` by :user:`glemaitre`.
- :class:`CrossValidationReport` now supports permutation feature importance through :func:`~CrossValidationReport.inspection.permutation_importance`. See :pr:`2370` by :user:`glemaitre`.
- :class:`ComparisonReport` now supports mean decrease in impurity (MDI) feature importance through :func:`~ComparisonReport.inspection.impurity_decrease`. See :pr:`2387` by :user:`auguste-probabl`.
- :class:`ConfusionMatrixDisplay` now supports ``threshold_value="all"`` to display confusion matrices for all available thresholds. See :pr:`2463` by :user:`glemaitre`.

Fixed
-----

- Tab completion now works correctly for reports in IPython. See :pr:`2427` by :user:`jeromedockes`.
- :class:`PredictionErrorDisplay` now supports multioutput regression models. See :pr:`2434` by :user:`GaetandeCast`.
- :class:`CoefficientsDisplay` now uses separate y-axes for each model when comparing models with different features. See :pr:`2403` by :user:`GaetandeCast`.
- Reports no longer crash when ``y`` is provided as a list or tuple. See :pr:`2433` by :user:`cakedev0`.
- :func:`skore.train_test_split` is now more robust to large inputs. See :pr:`2404` by :user:`cakedev0`.
- Data normalization in the data accessor is now more robust when converting various input formats to DataFrames. See :pr:`2440` by :user:`cakedev0`.
- Project names are now limited to 64 characters when interacting with Skore Hub. See :pr:`2454` by :user:`auguste-probabl`.
- An error is now raised earlier and more clearly when a report is pushed to Skore Hub without a ``pos_label``. See :pr:`2436` by :user:`glemaitre`.
- Empty project names are no longer allowed when interacting with Skore Hub. See :pr:`2453` by :user:`auguste-probabl`.
- :class:`pandas.Timestamp` objects no longer crash uploading a report to Skore Hub. See :pr:`2484` by :user:`thomass-dev`.
