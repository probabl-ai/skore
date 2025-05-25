.. _reporters:

====================================
Structuring data science experiments
====================================

.. currentmodule:: skore

When experimenting with data science, many tasks are repetitive across experiments or
projects. While the data exploration, transformation, and model architecture design
might require to innovate, the evaluation, inspection, and comparison of predictive
models are usually repetitive. However, these tasks require to develop substantial
amount of code and to store useful information. In itself, it is a challenge to get it
right.

`skore` provides a set of *reporters* that provide the following features:

- Provide only methods applicable to the task at hand
- Cache intermediate results to speed-up the exploring predictive models
- Produce data science artifacts with the least amount of code

Below, we present the different type of reporters that `skore` provides.

.. _estimator_report:

Reporter for a single estimator
-------------------------------

:class:`EstimatorReport` is the core reporter in `skore`. It is designed to take
a scikit-learn compatible estimator and some training and test data. The training
data is optional if the estimator is already fitted. The parameter `fit` in the
constructor gives full control on the fitting process. Omitting to provide part of
the data will reduce the amount of available methods when it comes to inspecting the
model. For instance, one will not be able to inspect the metrics of the model on the
test data if the test data is not provided.

Model evaluation
^^^^^^^^^^^^^^^^

:obj:`EstimatorReport.metrics`


Model inspection
^^^^^^^^^^^^^^^^


Caching mechanism
^^^^^^^^^^^^^^^^^

:ref:`sphx_glr_auto_examples_technical_details_plot_cache_mechanism.py`

.. _cross_validation_report:

Cross-validation estimator
--------------------------

.. comparison_report:

Comparison report
-----------------
