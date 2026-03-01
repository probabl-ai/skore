Report for a single estimator
=============================

.. currentmodule:: skore

The class :class:`EstimatorReport` provides a report allowing to inspect and
evaluate a scikit-learn estimator in an interactive way. The functionalities of the
report are accessible through accessors.

.. autosummary::
   :toctree: ../api/
   :template: class_with_accessors.rst

   EstimatorReport

.. rubric:: Methods

.. autosummary::
   :toctree: ../api/
   :template: class_methods_no_index.rst

   EstimatorReport.help
   EstimatorReport.cache_predictions
   EstimatorReport.clear_cache
   EstimatorReport.get_predictions

.. rubric:: Accessors

.. autosummary::
   :toctree: ../api/
   :template: autosummary/accessor.rst

   EstimatorReport.data
   EstimatorReport.inspection
   EstimatorReport.metrics

.. _estimator_data:

Data
----

The `data` accessor helps you to get insights about the dataset used to train and test
your estimator.

.. include:: ../api/EstimatorReport.data.inc

.. _estimator_metrics:

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of your
estimator.

.. include:: ../api/EstimatorReport.metrics.inc

Inspection
----------

The `inspection` accessor helps you inspect your model by e.g. evaluating the importance
of the features in your model.

.. include:: ../api/EstimatorReport.inspection.inc
