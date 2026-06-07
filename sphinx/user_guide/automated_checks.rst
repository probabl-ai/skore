.. _automated_checks:

================
Automated checks
================

`skore` provides automated checks for common model quality pitfalls.
Use :meth:`~skore.EstimatorReport.checks.summarize` to run checks and get a
a summary of the findings. Findings come in two severities:

- **issues** flag a concrete modeling problem to fix (e.g. overfitting);
- **tips** do not signal a defect on their own but invite caution, typically on
  the interpretation of a result (e.g. feature importance).

Each finding has:

- a short explanation,
- a stable check code,
- and a link to this page.

Checks can be muted per call with the `ignore` parameter of :meth:`~skore.EstimatorReport.checks.summarize`:

.. code-block:: python

    report.checks.summarize(ignore=["SKD001"])

You can also set a global ignore list with :attr:`~skore.configuration.ignore_checks`:

.. code-block:: python

    from skore import configuration
    configuration.ignore_checks = ["SKD001"]

Some checks are expensive because they require model refits or permutation
predictions (e.g. :ref:`SKD011 <skd011-golden-feature>` and
:ref:`SKD012 <skd012-useless-features>`). Those are tagged as *slow* and
can be skipped with `fast_mode=True`:

.. code-block:: python

    report.checks.summarize(fast_mode=True)

In fast mode, slow checks that are not yet in the cache are not run; cached
slow results from a previous call are still surfaced. The HTML representation
of a report uses fast mode so it never triggers an expensive computation.

For cross-validation reports, checks are run per split and then aggregated
at report level through :meth:`~skore.CrossValidationReport.checks.summarize`. An issue is
reported only when it appears in a strict majority of evaluated splits.

For comparison reports, :meth:`~skore.ComparisonReport.checks.summarize` builds a global
summary from each component report in the comparison. Issues are grouped by
component report and emitted as a single message.


.. _skd001-overfitting:

SKD001 - Potential overfitting
------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

`skore` compares train and test scores across the report's default predictive metrics
(timing metrics are excluded). A metric votes for overfitting when the train-favored
gap exceeds an adaptive threshold:

- **higher-is-better** metrics: ``train - test >= threshold``
- **lower-is-better** metrics: ``test - train >= threshold``

The threshold adapts to the scale of the scores:
``max(0.03, 0.10 * |reference|)`` where the reference is the train score for
higher-is-better metrics and the test score for lower-is-better metrics.
The floor of 0.03 prevents the threshold from vanishing on near-zero scores.

The check detects an issue when a **strict majority** of metrics vote for overfitting.

Why it matters
^^^^^^^^^^^^^^

A persistent train/test gap suggests the model has captured patterns specific to the
training data and may generalize poorly.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- simplify the model,
- regularize more strongly,
- improve feature engineering,
- use better validation protocols or more data.


.. _skd002-underfitting:

SKD002 - Potential underfitting
-------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

`skore` checks two conditions together across the report's default predictive metrics.
A metric votes for underfitting when **both** hold:

1. **Train and test scores are on par**: the absolute difference is within
   ``max(0.03, 0.05 * max(|train|, |test|))``.
2. **Neither score significantly outperforms a dummy baseline**: a score is considered
   significantly better than the baseline only when it exceeds
   ``max(0.01, 0.03 * |baseline|)``. The baseline is a ``DummyClassifier(strategy="prior")``
   for classification and a ``DummyRegressor(strategy="mean")`` for regression.

The check detects an issue when a **strict majority** of comparable metrics (those
present in both the estimator and dummy reports) vote for underfitting.

Why it matters
^^^^^^^^^^^^^^

When model performance is close to a naive baseline, the model is likely too simple,
under-trained, or using features that do not capture enough signal.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- increase model capacity,
- improve data representation and features,
- tune hyperparameters (see :ref:`SKD015 <skd015-hyperparameters-worth-tuning>`
  and :ref:`SKD016 <skd016-estimator-not-tuned>`),
- collect richer data if possible.


.. _skd003-inconsistent-performance:

SKD003 - Inconsistent performance across splits
-----------------------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

This check applies only to :class:`~skore.CrossValidationReport`.

`skore` examines each split's test scores across the report's default predictive metrics
(timing metrics are excluded). For every metric, a `modified Z-score
<https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation>`_
based on the Median Absolute Deviation (MAD) is computed:

.. math::

   z_i = 0.6745 \times \frac{x_i - \widetilde{x}}{\text{MAD}}

It is a version of the Z-score that is more robust to extreme values and does not make
assumptions about the distribution of the data.
A split is flagged as an outlier for a given metric when :math:`|z_i| > 3` which is
analogous to being outside 3 standard deviations from the mean.

A split is considered inconsistent when a **strict majority** of metrics flag it as an
outlier. The check reports an issue if at least one split is labeled inconsistent.

Why it matters
^^^^^^^^^^^^^^

When one or more splits perform very differently from the rest, the cross-validation
estimate becomes unreliable. The anomaly may reveal data leakage, uneven class
distributions across splits, or a model that is sensitive to specific data splits.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- use stratified or grouped cross-validation to ensure a more even split,
- investigate whether the outlier split contains a different data distribution,
- check for data leakage or temporal effects,
- increase the size of the dataset to improve stability.


.. _skd004-high-class-imbalance:

SKD004 - High class imbalance
------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

This check applies only to binary classification tasks.

`skore` counts the occurrences of each class across the train and test sets. The check
detects an issue when the most frequent class represents more than **80 %** of the
dataset.

Why it matters
^^^^^^^^^^^^^^

When one class dominates the dataset, a model can achieve high accuracy simply by
constantly predicting the majority class. Accuracy alone becomes a misleading performance
indicator, and the model may fail to detect the minority class entirely.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- use metrics that account for imbalance (precision, recall, F1, ROC AUC),
- resample the dataset (oversampling the minority or undersampling the majority),
- use class weights in the estimator,
- collect more data for the minority class if possible.


.. _skd005-underrepresented-classes:

SKD005 - Underrepresented classes
---------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

This check applies only to multiclass classification tasks.

`skore` counts the occurrences of each class across the train and test sets. The check
detects an issue when one or more classes each represent less than **10 %** of the
dataset.

Why it matters
^^^^^^^^^^^^^^

When some classes are severely underrepresented, the model may never learn to
distinguish them reliably. Overall accuracy can look acceptable while per-class
performance on the rare classes remains poor.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- use per-class metrics (precision, recall, F1 per class) to monitor all classes,
- resample the dataset (oversampling rare classes or undersampling frequent ones),
- use class weights in the estimator,
- collect more data for the underrepresented classes if possible.


.. _skd006-unscaled-coefficients:

SKD006 - Coefficient interpretation
------------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

This check applies only to linear estimators that expose a ``coef_`` attribute.

`skore` concatenates the train and test inputs and computes the per-feature
standard deviation. The check emits one of two tips depending on the result:

- **Features are on different scales** (standard deviations are not close to
  each other): coefficient magnitudes are not directly comparable across
  features.
- **Features are on the same scale**: coefficient magnitudes are comparable
  as feature importance, but they are no longer interpretable in the original
  feature units.

Why it matters
^^^^^^^^^^^^^^

The magnitude of a linear model's coefficients depends on the scale of each input
feature. When features live on different scales, comparing raw coefficients as a
measure of feature importance is misleading: a large coefficient may only reflect
a small-scale feature, not a strong effect. Standardizing the inputs puts all
coefficients on a common footing and makes them directly comparable.

Conversely, when features have been standardized, coefficients express changes per
standard deviation rather than per original unit. Statements like "an increase of
1 year in AGE means a decrease of 0.03 $/hour" lose their meaning because the
natural units have been scaled away.

Read more about this in `the scikit-learn documentation
<https://scikit-learn.org/dev/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html#interpreting-coefficients-scale-matters>`__.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- standardize the inputs (e.g. wrap the estimator in a pipeline with
  :class:`~sklearn.preprocessing.StandardScaler`) to make coefficients
  comparable,
- when features are not standardized, multiply each coefficient by the feature's
  standard deviation to make them comparable,
- otherwise, interpret coefficient magnitudes only relative to the feature's own
  scale, or rely on scale-invariant feature-importance methods such as
  :class:`~skore.PermutationImportanceDisplay`.


.. _skd007-mdi-cardinality-bias:

SKD007 - MDI feature importance is biased for high-cardinality features
--------------------------------------------------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

This check applies only to estimators that expose a ``feature_importances_``
attribute (tree-based models such as random forests and gradient boosting).

`skore` counts the number of unique values in each feature column.
A feature is considered high-cardinality when its number of unique values
exceeds **50 % of the number of samples**. The check emits a tip when at
least one such feature exists.

Why it matters
^^^^^^^^^^^^^^

The default feature importance reported by tree ensembles is Mean Decrease in
Impurity (MDI). MDI is computed from the training set and is biased toward
high-cardinality and continuous features: because those features offer more
candidate split points, trees tend to select them more often, inflating their
apparent importance even when they carry no real predictive signal.

Read more about this in `the scikit-learn documentation
<https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html>`__.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- use permutation importance (:class:`~skore.PermutationImportanceDisplay`)
  instead of MDI: it measures importance on the test set and is not biased by
  cardinality,
- if MDI is needed, be aware of its limitations and cross-check with
  permutation importance or drop-column importance.


.. _skd008-correlated-features:

SKD008 - Highly correlated input features
-----------------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

`skore` computes the pairwise `Spearman rank correlation
<https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_
between all numeric features. The check emits an issue when at least one pair
of features has an absolute correlation above **0.9**
(``|rho| > 0.9``).

Why it matters
^^^^^^^^^^^^^^

Highly correlated features carry largely redundant information. This redundancy
can cause two problems:

- **Unreliable feature importance**: when two features are highly correlated,
  importance methods (MDI, permutation importance, linear-model coefficients)
  can arbitrarily split credit between them. A genuinely important feature may
  receive near-zero importance simply because a correlated partner captured
  most of the signal, and the estimates become highly variable across data
  perturbations. Read more `here
  <https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html>`_.
- **Wasted resources and overfitting risk**: redundant features increase memory
  usage and computation time without adding information. Removing them can speed
  up training and, by reducing the effective dimensionality, may also reduce
  overfitting.
- **Degenerate linear models**: near-perfect collinearity makes the design
  matrix ill-conditioned, inflating coefficient variance and making
  least-squares estimates numerically unstable.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- remove or combine redundant features before fitting (e.g. drop one of each
  highly correlated pair, or use dimensionality reduction),
- use regularization (Lasso, ElasticNet) to let the model select among
  correlated features,
- group correlated features together before inspecting feature importance.

.. _skd009-worse-than-baseline:

SKD009 - Model worse than baseline
----------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

`skore` builds a strong baseline made of :func:`skrub.tabular_pipeline` wrapped around
:class:`~sklearn.ensemble.HistGradientBoostingClassifier` for classification tasks or
:class:`~sklearn.ensemble.HistGradientBoostingRegressor` for regression tasks. The baseline
is trained on the same train data as the report's estimator and is evaluated on the same
test set.

For each of the report's default predictive metrics (timing metrics are excluded), a
metric votes for the issue when the report is **not significantly better** than the
baseline. A score is considered significantly better only when its gap to the baseline
exceeds ``max(0.01, 0.05 * |baseline|)``.

The check detects an issue when a **strict majority** of comparable metrics vote.

Why it matters
^^^^^^^^^^^^^^

If the model does not match or beat a sensible off-the-shelf baseline, the modeling
effort may not be worth its complexity: a simpler, well-tuned default could deliver the
same quality with less risk of overfitting or maintenance burden.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- revisit feature engineering and preprocessing,
- tune the model's hyperparameters (see
  :ref:`SKD015 <skd015-hyperparameters-worth-tuning>` and
  :ref:`SKD016 <skd016-estimator-not-tuned>`),
- check whether the model family is appropriate for the data,
- consider switching to a stronger default such as
  :class:`~sklearn.ensemble.HistGradientBoostingClassifier` /
  :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.


.. _skd010-slower-than-baseline:

SKD010 - Model slower than baseline
-----------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

`skore` builds a fast linear baseline made of :func:`skrub.tabular_pipeline` wrapped
around :class:`~sklearn.linear_model.LogisticRegression` for classification tasks or
:class:`~sklearn.linear_model.RidgeCV` for regression tasks. The baseline is trained on
the same train data as the report's estimator and is evaluated on the same test set.

The check first compares fit times: it triggers only when the report's ``fit_time_`` is
at least **2x** the baseline's fit time and the absolute gap is at least 0.05 seconds
(the floor avoids spurious results on very fast fits).

Then, like :ref:`SKD009 <skd009-worse-than-baseline>`, each default predictive metric
votes for the issue when the report is **not significantly better** than the baseline on
the test set, with the same ``max(0.01, 0.05 * |baseline|)`` adaptive threshold.

The check detects an issue when the slowness gate holds **and** a strict majority of
comparable metrics vote.

Why it matters
^^^^^^^^^^^^^^

A model that is much slower than a simple linear baseline but does not deliver
better predictive quality wastes compute and engineering time. The added complexity
is rarely justified when a fast linear model already captures the signal.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- switch to the fast linear baseline when its quality is sufficient,
- reduce the model's complexity (e.g. fewer trees, lower depth, smaller
  hyperparameter grids),
- profile fit time to understand the dominant cost,
- check that the input pipeline (encoding, scaling) is not the actual bottleneck.


.. _skd011-golden-feature:

SKD011 - Golden feature
-----------------------

This check is *slow*: it requires fitting one model per feature. Skip it with
``fast_mode=True``.

How it is detected
^^^^^^^^^^^^^^^^^^

For each input feature, `skore` clones the report's estimator, refits it on
that single feature, and scores it on the test set. A feature is considered as
*golden* when its single-feature scores are close to the full model's scores within
an adaptive threshold (``max(0.03, 0.10 * |full_score|)``) on a **strict
majority** of the report's default predictive metrics (timing metrics
excluded).

The check only runs when the report has at least two features.

Why it matters
^^^^^^^^^^^^^^

A single feature that already captures (almost) all the predictive signal is
often a symptom of data leakage (e.g. a column that encodes the target) or
of an over-reliance on one feature that could hurt robustness in production
if the feature's distribution changes.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- audit the suspect feature for leakage (is it derived from the target or
  from data that would not be available at inference time?),
- compare predictive performance with and without the feature,
- collect or engineer additional features so the model is less dependent on
  a single one.


.. _skd012-useless-features:

SKD012 - Useless features
-------------------------

This check is *slow*: it computes permutation importance. Skip it with
``fast_mode=True``.

How it is detected
^^^^^^^^^^^^^^^^^^

`skore` uses permutation importance on the test set to assess each feature's
contribution. A feature is flagged as *useless* when:

- its mean importance is **below 1e-3** (catches negligible or negative
  values regardless of variance), or
- its importance interval ``[mean - std, mean + std]`` **contains zero**.

Why it matters
^^^^^^^^^^^^^^

Features whose permutation importance is negligible contribute little
or no measurable signal: shuffling their values does not degrade the model's
score. They are good candidates for dropping, which can simplify the model
and reduce overfitting risk.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- review the flagged features and consider dropping them,
- refit the model on the reduced feature set and verify performance is
  preserved,
- if a flagged feature is expected to matter, investigate whether the model
  is too simple or the feature is poorly encoded.

.. note::

   In the presence of correlated features, permutation importance can be near
   zero even for predictive features (see `Permutation Importance with
   Multicollinear or Correlated Features
   <https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html>`_).
   When input features are correlated, drop zero-importance features one by
   one — e.g. with scikit-learn's :class:`~sklearn.feature_selection.RFECV` —
   rather than all at once.


.. _skd013-train-test-time-overlap:

SKD013 - Train-test overlap in time series
------------------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^

This check only applies when both ``X_train`` and ``X_test`` are pandas
DataFrames. For every column with a datetime dtype, `skore` checks whether
the latest training timestamp is greater than or equal to the earliest test
timestamp. When that is the case for any column, the check reports an issue
and lists the impacted columns.

Why it matters
^^^^^^^^^^^^^^

If future points are present in the training set, the evaluation score
becomes optimistic because the model gets to see information that would
not be available at inference time. This is a common pitfall when a
time-indexed dataset is shuffled before splitting.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- use a time-based splitter such as
  :class:`~sklearn.model_selection.TimeSeriesSplit` or similar.

.. _skd014-hyperparams-at-search-edge:

SKD014 - Hyperparameters at search edge
---------------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^
The check runs only when the report's estimator is a fitted
:class:`~sklearn.model_selection.BaseSearchCV` object (e.g.
:class:`~sklearn.model_selection.GridSearchCV` or
:class:`~sklearn.model_selection.RandomizedSearchCV`).

Only numeric hyperparameters are considered. For each one, `skore` flags an
issue when ``best_params_`` equals the **minimum or maximum** distinct value
tried during the search. Order in ``param_grid`` does not matter.

Why it matters
^^^^^^^^^^^^^^
When the best hyperparameters sit on the boundary of what was actually explored,
the true optimum may lie outside the searched range. Extending the grid or
distributions is often worthwhile.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- extend ``param_grid`` or ``param_distributions`` beyond the flagged bounds,
- for :class:`~sklearn.model_selection.RandomizedSearchCV`, increase ``n_iter``
  and sample from a wider range,
- if :ref:`SKD015 <skd015-hyperparameters-worth-tuning>` also fires, address
  both together: the search space is too narrow on at least one axis and is
  also missing recommended axes entirely.

.. _skd015-hyperparameters-worth-tuning:

SKD015 - Hyperparameters worth tuning
-------------------------------------

How it is detected
^^^^^^^^^^^^^^^^^^
The check runs only when the report's estimator is a fitted
:class:`~sklearn.model_selection.BaseSearchCV` object. The searched parameters
are compared to a curated table of hyperparameters considered most impactful in
the tuning literature (Probst, Boulesteix & Bischl 2019; van Rijn & Hutter 2018).

When the search wraps a :class:`~sklearn.pipeline.Pipeline`, every step whose
class is in the table is checked independently, regardless of whether the search
currently tunes any of its parameters. Recommended axes that play the same role
(e.g. ``max_depth`` and ``min_samples_leaf`` for tree complexity) are collapsed
to a single suggestion.

Why it matters
^^^^^^^^^^^^^^
Not tuning the most impactful hyperparameters leaves performance on the table.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- add the suggested parameters to ``param_grid`` or ``param_distributions``.

.. _skd016-estimator-not-tuned:

SKD016 - Estimator not tuned
----------------------------

How it is detected
^^^^^^^^^^^^^^^^^^
The check runs on plain estimators and :class:`~sklearn.pipeline.Pipeline`
object.

For every step whose class is in the recommendation table, `skore` lists the
init params that differ from their scikit-learn default. Infrastructure params
that do not affect the learned model (``random_state``, ``n_jobs``, ``verbose``,
``warm_start``, ``class_weight``, ``copy`` / ``copy_X``, ``cache_size``) are
ignored. When every remaining param of a step is still at its default, the
check reports a tip suggesting the recommended tuning axes for that class.

Why it matters
^^^^^^^^^^^^^^
Default hyperparameters are rarely optimal, and an estimator left fully at
defaults usually under-performs a tuned one.

How to reduce the risk
^^^^^^^^^^^^^^^^^^^^^^

- wrap the estimator in :class:`~sklearn.model_selection.GridSearchCV` or
  :class:`~sklearn.model_selection.RandomizedSearchCV` over the suggested
  parameters,
- or set sensible non-default values manually.
