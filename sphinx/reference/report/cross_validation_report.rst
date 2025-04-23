Report for a cross-validation of an estimator
=============================================

.. currentmodule:: skore

The class :class:`CrossValidationReport` performs cross-validation and provides a report
to inspect and evaluate a scikit-learn estimator in an interactive way. The
functionalities of the report are exposed through accessors.

.. autoclass:: CrossValidationReport
   :members:
   :exclude-members: metrics
   
   .. rubric:: Metrics
   .. autoattribute:: metrics
      :noindex:

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of your
estimator across cross-validation folds.

.. removed manually specifying methods to avoid duplicates.
