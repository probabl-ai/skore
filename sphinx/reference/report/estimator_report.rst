Report for a single estimator
=============================

.. currentmodule:: skore

The class :class:`EstimatorReport` provides a report allowing to inspect and
evaluate a scikit-learn estimator in an interactive way. The functionalities of the
report are accessible through accessors.

.. autosummary::
   :toctree: ../api/
   :template: base.rst

   EstimatorReport

.. rubric:: Methods

.. removed manually specifying methods to avoid duplicates.

.. rubric:: Metrics

.. autosummary::
   :toctree: ../api/
   :template: autosummary/accessor.rst

   EstimatorReport.metrics

.. rubric:: Feature importance

.. autosummary::
   :toctree: ../api/
   :template: autosummary/accessor.rst

   EstimatorReport.feature_importance

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of your
estimator.

.. removed manually specifying methods to avoid duplicates.

Feature importance
------------------

The `feature_importance` accessor helps you to evaluate the importance of the features
used to train your estimator.

.. removed manually specifying methods to avoid duplicates.
