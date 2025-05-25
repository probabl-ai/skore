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
first available method with the `skore` display is a `plot` method. It allows to
show graphically the information contained in the display. It can be recalled as many
time as you want and does not modify the display object and does not require heavy
computation.

.. plot::
    :context: close-figs
    :align: center

    disp.plot()

The `plot` method accepts parameters to tweak the rendering of the display. For
instance, we can tweak the line style of the chance level line:

.. plot::
    :context: close-figs
    :align: center

    disp.plot(
        chance_level_kwargs=dict(
            linestyle="-", linewidth=5, color="tab:purple"
        )
    )

While it can be cumbersome to always pass the parameters at each call to `plot`, a
method `set_style` is available to persist some style settings.

.. plot::
    :context: close-figs
    :align: center

    disp.set_style(
        chance_level_kwargs=dict(linestyle="-", linewidth=5, color="tab:purple")
    )
    disp.plot()

Now, any subsequent call to `plot` will use the style settings set by `set_style`.
