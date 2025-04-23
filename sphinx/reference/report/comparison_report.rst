Report for a comparison of :class:`EstimatorReport`
===================================================

.. currentmodule:: skore

The class :class:`ComparisonReport` provides a report allowing to compare :class:`EstimatorReport` instances in an interactive way. The functionalities of the report are accessible through accessors.

.. autoclass:: ComparisonReport
   :members:
   :exclude-members: metrics
   
   .. rubric:: Metrics
   .. autoattribute:: metrics
      :noindex:

Metrics
-------

The `metrics` accessor helps you to evaluate the statistical performance of the
compared estimators. In addition, we provide a sub-accessor `plot`, to
get the common performance metric representations.

.. removed manually specifying methods to avoid duplicates.
