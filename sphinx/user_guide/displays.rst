.. _displays:

=========================================
Visualization via the `skore` display API
=========================================

.. currentmodule:: skore

`skore` provides a family of objects that we call displays. All displays follow the
common API defined by the :class:`Display` protocol. As a user, you get a display by
interacting with a reporter. Let's provide an example:

.. plot::
    :context: close-figs
    :align: center

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from skore import CrossValidationReport

    X, y = make_classification(
        n_samples=10_000,
        n_classes=3,
        class_sep=0.3,
        n_clusters_per_class=1,
        random_state=42,
    )
    report = CrossValidationReport(LogisticRegression(), X, y)
    disp = report.metrics.roc()
    disp.plot()

The :meth:`EstimatorReport.metrics.roc` creates a :class:`RocCurveDisplay` object. The
first available method with the `skore` display is a `plot` method. It shows graphically
the information contained in the display. Call it as many times as you want - it does
not modify the display object nor require heavy computation.

.. plot::
    :context: close-figs
    :align: center

    disp.plot()

The `plot` method accepts parameters to tweak the rendering of the display. For
instance, customize the appearance of the chance level:

.. plot::
    :context: close-figs
    :align: center

    disp.plot(
        chance_level_kwargs=dict(
            linestyle="-", linewidth=5, color="tab:purple"
        )
    )

To avoid passing parameters at each call to `plot`, use the `set_style` method to
persist style settings.

.. plot::
    :context: close-figs
    :align: center

    disp.set_style(
        chance_level_kwargs=dict(linestyle="-", linewidth=5, color="tab:purple")
    )
    disp.plot()

Any subsequent call to `plot` uses the style settings set by `set_style`.
