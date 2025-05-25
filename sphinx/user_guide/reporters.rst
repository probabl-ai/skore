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

:obj:`EstimatorReport.metrics` is the entry point that provides methods to evaluate
the statistical and performance metrics of the predictive model. This accessor provides
two types of methods: (i) methods that return some metrics and (ii) methods that
return a `skore` display object.

Before diving into the details of these methods, we first discuss the parameters they share.
`data_source` is a parameter that allows specifying the data
to use to compute the metrics. It can be set to `train` or `test` in which case we
rely on the data provided to the constructor. In addition, `data_source` can be set
to `X_y` and it allows to pass a new dataset using the parameters `X` and `y`. It is
useful when you want to compare different models on a new left-out dataset.

While there is individual methods to compute each metric specific to the problem at
hand, we provide the :class:`EstimatorReport.metrics.report_metrics` method that
allows to aggregate metrics in a single dataframe. By default, a set of metrics is
computed based on the type of target variable. However, you can specify by passing
the metrics you want to compute to the `scoring` parameter. We accept different types:
(i) some strings that corresponds to scikit-learn scorer names or a built-in `skore`
metric name, (ii) a callable or a (iii) scikit-learn scorer constructed with
:func:`sklearn.metrics.make_scorer`.

Refer to the :ref:`estimator_report_metrics` section for more details on all the
available metrics in `skore`.

The second type of methods provided by :obj:`EstimatorReport.metrics` are methods
that return a `skore` display object. They have a common API as well. They expose
two methods: (i) `plot` that allows to plot graphically the information contained
in the display and (ii) `set_style` that allows to set some graphical settings instead
of passing them to the `plot` method at each call.

Refer to the :ref:`displays` section for more details regarding the `skore` display
API.

Caching mechanism
^^^^^^^^^^^^^^^^^

:ref:`sphx_glr_auto_examples_technical_details_plot_cache_mechanism.py`

.. _cross_validation_report:

Cross-validation estimator
--------------------------

.. comparison_report:

Comparison report
-----------------
