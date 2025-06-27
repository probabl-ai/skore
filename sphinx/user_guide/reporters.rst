.. _reporters:

====================================
Structuring data science experiments
====================================

.. currentmodule:: skore

When experimenting with data science, many tasks are repetitive across experiments or
projects. While the data exploration, transformation, and model architecture design
might require innovation, the evaluation, inspection, and comparison of predictive
models are usually repetitive. However, these tasks require developing substantial
amounts of code and storing useful information. In itself, it is a challenge to get it
right.

`skore` provides a set of *reporters* that provide the following features:

- Provide only methods applicable to the task at hand
- Cache intermediate results to speed up exploring predictive models
- Produce data science artifacts with the least amount of code

Below, we present the different types of reporters that `skore` provides.

.. _estimator_report:

Reporter for a single estimator
-------------------------------

:class:`EstimatorReport` is the core reporter in `skore`. It is designed to take a
scikit-learn compatible estimator and some training and test data. The training data is
optional if the estimator is already fitted. The parameter `fit` in the constructor
gives full control over the fitting process. Omitting part of the data reduces the
number of available methods when inspecting the model. For instance, you cannot inspect
the metrics of the model on the test data if you do not provide the test data.

Model evaluation
^^^^^^^^^^^^^^^^

:obj:`EstimatorReport.metrics` is the entry point that provides methods to evaluate the
statistical and performance metrics of the predictive model. This accessor provides two
types of methods: (i) methods that return some metrics and (ii) methods that return a
`skore` :class:`Display` object.

Before diving into the details of these methods, we first discuss the parameters they
share. `data_source` is a parameter that specifies the data to use to compute the
metrics. Set it to `train` or `test` to rely on the data provided to the constructor. In
addition, set `data_source` to `X_y` to pass a new dataset using the parameters `X` and
`y`. This is useful when you want to compare different models on a new left-out dataset.

There are individual methods to compute each metric specific to the problem at hand.
They return usual python objects such as floats, integers, or dictionaries.

The second type of methods provided by :obj:`EstimatorReport.metrics` are methods that
return a :class:`~skore.Display` object. They have a common API as well. They expose
three methods:
(i) `plot` that plots graphically the information contained in the display,
(ii) `set_style` that sets some graphical settings instead of passing them to the `plot`
method at each call.
(iii) `frame` that returns a `pandas.DataFrame` with the information contained in the
display.

We provide the :class:`EstimatorReport.metrics.summarize` method that aggregates metrics
in a single dataframe, available through a :class:`~skore.Display`. By default, a set of
metrics is computed based on the type of target variable (e.g. classification or
regression). Nevertheless, you can specify the metrics you want to compute thanks to the
`scoring` parameter. We accept different types: (i) some strings that correspond to
scikit-learn scorer names or a built-in `skore` metric name, (ii) a callable or a (iii)
scikit-learn scorer constructed with :func:`sklearn.metrics.make_scorer`.

Refer to the :ref:`displays` section for more details regarding the `skore` display
API. Refer to the :ref:`estimator_metrics` section for more details on all the
available metrics in `skore`.

Caching mechanism
^^^^^^^^^^^^^^^^^

:class:`EstimatorReport` comes together with a caching mechanism that stores
intermediate information that is expensive to compute such as predictions. It
efficiently re-uses this information when recomputing the same metric or a metric
requiring the same intermediate information.

We expose three methods to interact with the cache:

- :meth:`EstimatorReport.cache_predictions` to cache the predictions of the estimator
  without awaiting the computation when calling the evaluation metrics.
- :meth:`EstimatorReport.clear_cache` to clear the cache.
- :meth:`EstimatorReport.get_predictions` to get the predictions from the cache or
  compute them if they are not in the cache.

.. note::
    The current implementation of the caching mechanism happens in-memory. It is
    therefore not persisted between sessions, apart from loading an
    :class:`EstimatorReport` from a :class:`Project`. Refer to the following
    section :ref:`project` for more details.

Refer to the example entitled
:ref:`sphx_glr_auto_examples_technical_details_plot_cache_mechanism.py` to get a
detailed view of the caching mechanism.

.. _cross_validation_report:

Cross-validation estimator
--------------------------

:class:`CrossValidationReport` has a similar API to :class:`EstimatorReport`. The main
difference is in the initialization. It accepts an estimator, a dataset (i.e. `X` and
`y`) and a cross-validation strategy. Internally, the dataset is split according to the
cross-validation strategy and an estimator report is created for each split. Therefore,
a :class:`CrossValidationReport` is a collection of :class:`EstimatorReport` instances,
available through the :obj:`CrossValidationReport.estimator_reports_` attribute.

For metrics and displays, the same API is exposed with an extra
parameter, `aggregate`, to aggregate the metrics across the splits.

The :class:`CrossValidationReport` also comes with a caching mechanism by leveraging
the :class:`EstimatorReport` caching mechanism and exposes the same methods.

Refer to the :ref:`cross_validation_metrics` section for more details on the
metrics available in `skore` for cross-validation.

.. _comparison_report:

Comparison report
-----------------

To compare the performance of different predictive models, `skore` provides the
:class:`ComparisonReport`.

:class:`ComparisonReport` takes a list (or a dictionary) of :class:`EstimatorReport` or
:class:`CrossValidationReport` instances. It then provides methods to compare the
performance of the different models.

The reports to be compared must share the same test target, for the comparison to make
sense. The reports may not share the same training data or target, therefore the
:class:`ComparisonReport` can be used to diagnose the performance of a new model,
compared to a historical one in production for instance. They may also not share the
same testing X, to remove the constraints of having consistent pipeline input.
This comparison of testing target is done thanks to hashing of the table.

The caching mechanism is also available and exposes the same methods.

Refer to the :ref:`cross_validation_metrics` section for more details on the
metrics available in `skore` for comparison.
