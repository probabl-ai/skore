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

- Help menus accessed through ``help()`` methods now render as interactive HTML in Jupyter notebooks and IPython environments. See `#2316 <https://github.com/probabl-ai/skore/pull/2316>`__ by `@glemaitre <https://github.com/glemaitre>`__

Changed
-------

- **Breaking:** Reports no longer accept clustering models and only support supervised learning tasks (classification and regression). See `#2489 <https://github.com/probabl-ai/skore/pull/2489>`__ by `@GaetandeCast <https://github.com/GaetandeCast>`__
- **Breaking:** The ``mode`` parameter is now required when creating a ``Project`` in "hub" mode. See `#2401 <https://github.com/probabl-ai/skore/pull/2401>`__ by `@thomass-dev <https://github.com/thomass-dev>`__
- DataFrame column names in reports now consistently use singular form (e.g., "model" instead of "models"). See `#2392 <https://github.com/probabl-ai/skore/pull/2392>`__ by `@Sharkyii <https://github.com/Sharkyii>`__
- Exceptions raised during the ``CrossValidationReport`` fitting process are no longer caught by skore. See `#2462 <https://github.com/probabl-ai/skore/pull/2462>`__ by `@glemaitre <https://github.com/glemaitre>`__

Added
-----

- The documentation now includes a short example illustrating local usage of ``skore.Project``. See `#2481 <https://github.com/probabl-ai/skore/pull/2481>`__ by `@glemaitre <https://github.com/glemaitre>`__
- The documentation now includes a short example explaining the design philosophy of the skore API. See `#2480 <https://github.com/probabl-ai/skore/pull/2480>`__ by `@glemaitre <https://github.com/glemaitre>`__
- The report URL is now printed after uploading to Skore Hub. See `#2488 <https://github.com/probabl-ai/skore/pull/2488>`__ by `@rouk1 <https://github.com/rouk1>`__
- Reports now support setting ``pos_label`` after initialization, enabling users to change the positive class for binary classification metrics. See `#2438 <https://github.com/probabl-ai/skore/pull/2438>`__ by `@glemaitre <https://github.com/glemaitre>`__
- ``CrossValidationReport`` now supports permutation feature importance through :func:`~CrossValidationReport.inspection.permutation_importance`. See `#2370 <https://github.com/probabl-ai/skore/pull/2370>`__ by `@glemaitre <https://github.com/glemaitre>`__
- ``ComparisonReport`` now supports mean decrease in impurity (MDI) feature importance through :func:`~ComparisonReport.inspection.impurity_decrease`. See `#2387 <https://github.com/probabl-ai/skore/pull/2387>`__ by `@auguste-probabl <https://github.com/auguste-probabl>`__
- ``ConfusionMatrixDisplay`` now supports ``threshold_value="all"`` to display confusion matrices for all available thresholds. See `#2463 <https://github.com/probabl-ai/skore/pull/2463>`__ by `@glemaitre <https://github.com/glemaitre>`__

Removed
-------

Fixed
-----

- Tab completion now works correctly for reports in IPython. See `#2427 <https://github.com/probabl-ai/skore/pull/2427>`__ by `@jeromedockes <https://github.com/jeromedockes>`__
- ``PredictionErrorDisplay`` now supports multioutput regression models. See `#2434 <https://github.com/probabl-ai/skore/pull/2434>`__ by `@GaetandeCast <https://github.com/GaetandeCast>`__
- ``CoefficientsDisplay`` now uses separate y-axes for each model when comparing models with different features. See `#2403 <https://github.com/probabl-ai/skore/pull/2403>`__ by `@GaetandeCast <https://github.com/GaetandeCast>`__
- Reports no longer crash when ``y`` is provided as a list or tuple. See `#2433 <https://github.com/probabl-ai/skore/pull/2433>`__ by `@cakedev0 <https://github.com/cakedev0>`__
- ``train_test_split`` is now more robust to large inputs. See `#2404 <https://github.com/probabl-ai/skore/pull/2404>`__ by `@cakedev0 <https://github.com/cakedev0>`__
- Data normalization in the data accessor is now more robust when converting various input formats to DataFrames. See `#2440 <https://github.com/probabl-ai/skore/pull/2440>`__ by `@cakedev0 <https://github.com/cakedev0>`__
- Project names are now limited to 64 characters when interacting with Skore Hub. See `#2454 <https://github.com/probabl-ai/skore/pull/2454>`__ by `@auguste-probabl <https://github.com/auguste-probabl>`__
- An error is now raised earlier and more clearly when a report is pushed to Skore Hub without a ``pos_label``. See `#2436 <https://github.com/probabl-ai/skore/pull/2436>`__ by `@glemaitre <https://github.com/glemaitre>`__
- Empty project names are no longer allowed when interacting with Skore Hub. See `#2453 <https://github.com/probabl-ai/skore/pull/2453>`__ by `@auguste-probabl <https://github.com/auguste-probabl>`__
- ``pandas.Timestamp`` objects no longer crash uploading a report to Skore Hub. See `#2484 <https://github.com/probabl-ai/skore/pull/2484>`__ by `@thomass-dev <https://github.com/thomass-dev>`__
