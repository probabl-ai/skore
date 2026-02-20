.. _changes:

=========
Changelog
=========

.. currentmodule:: skore

Unreleased
==========

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

`0.13.0`_ (2026-02-25)
======================

.. _0.13.0: https://github.com/probabl-ai/skore/compare/skore/0.12.0...skore/0.13.0

Release highlights
------------------

- Help menus accessed through ``help()`` methods now render as interactive HTML in Jupyter notebooks and IPython environments. See :issue:`2316` by :user:`glemaitre`

Changed
-------

- **Breaking:** Reports no longer accept clustering models and only support supervised learning tasks (classification and regression). See :issue:`2489` by :user:`GaetandeCast`
- **Breaking:** The ``mode`` parameter is now required when creating a :class:`Project` in "hub" mode. See :issue:`2401` by :user:`thomass-dev`
- DataFrame column names in reports now consistently use singular form (e.g., "model" instead of "models"). See :issue:`2392` by :user:`Sharkyii`
- Exceptions raised during the :class:`CrossValidationReport` fitting process are no longer caught by skore. See :issue:`2462` by :user:`glemaitre`

Added
-----

- The documentation now includes a short example illustrating local usage of :class:`skore.Project`. See :issue:`2481` by :user:`glemaitre`
- The documentation now includes a short example explaining the design philosophy of the skore API. See :issue:`2480` by :user:`glemaitre`
- The report URL is now printed after uploading to Skore Hub. See :issue:`2488` by :user:`rouk1`
- Reports now support setting ``pos_label`` after initialization, enabling users to change the positive class for binary classification metrics. See :issue:`2438` by :user:`glemaitre`
- :class:`CrossValidationReport` now supports permutation feature importance through :func:`~CrossValidationReport.inspection.permutation_importance`. See :issue:`2370` by :user:`glemaitre`
- :class:`ComparisonReport` now supports mean decrease in impurity (MDI) feature importance through :func:`~ComparisonReport.inspection.impurity_decrease`. See :issue:`2387` by :user:`auguste-probabl`
- :class:`ConfusionMatrixDisplay` now supports ``threshold_value="all"`` to display confusion matrices for all available thresholds. See :issue:`2463` by :user:`glemaitre`

Removed
-------

Fixed
-----

- Tab completion now works correctly for reports in IPython. See :issue:`2427` by :user:`jeromedockes`
- :class:`PredictionErrorDisplay` now supports multioutput regression models. See :issue:`2434` by :user:`GaetandeCast`
- :class:`CoefficientsDisplay` now uses separate y-axes for each model when comparing models with different features. See :issue:`2403` by :user:`GaetandeCast`
- Reports no longer crash when ``y`` is provided as a list or tuple. See :issue:`2433` by :user:`cakedev0`
- :func:`skore.train_test_split` is now more robust to large inputs. See :issue:`2404` by :user:`cakedev0`
- Data normalization in the data accessor is now more robust when converting various input formats to DataFrames. See :issue:`2440` by :user:`cakedev0`
- Project names are now limited to 64 characters when interacting with Skore Hub. See :issue:`2454` by :user:`auguste-probabl`
- An error is now raised earlier and more clearly when a report is pushed to Skore Hub without a ``pos_label``. See :issue:`2436` by :user:`glemaitre`
- Empty project names are no longer allowed when interacting with Skore Hub. See :issue:`2453` by :user:`auguste-probabl`
- :class:`pandas.Timestamp` objects no longer crash uploading a report to Skore Hub. See :issue:`2484` by :user:`thomass-dev`
