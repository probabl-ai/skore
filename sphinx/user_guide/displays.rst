.. _displays:

======================================
From evaluation results to clear views
======================================

.. currentmodule:: skore

Once you have fitted a model and scored it, the next step is almost always the same in
spirit and tedious in practice: you need **figures and tables**—curves, matrices,
residual plots, summaries of the data—that explain **how well** the model behaves and
**why** it looks that way. Doing that by hand means wiring predictions, labels, and
metrics into matplotlib or seaborn over and over, with slightly different function
signatures every time, recomputing expensive steps when you tweak a plot, and juggling
notebook “magic” versus scripts that must call ``show()`` explicitly. When you finally
need the **numbers** behind a chart—for a report, a dashboard, or a statistical
test—you often end up duplicating logic or scraping axes objects.

Skore’s **display** objects target that gap. A display is a small, dedicated object you
obtain from a **report** (for example via accessors such as ``report.metrics`` or
``report.data``). It already holds the computed quantities needed for a given diagnostic
view. You are not meant to construct displays from scratch; they are the visualization
layer that sits on top of the same evaluation you used to build the report, so the story
from metrics to plot stays in one place.

All displays share the same idea of a **stable API**, formalized by the :class:`Display`
protocol, so learning one display teaches you the others:

- ``plot`` builds a :class:`matplotlib.figure.Figure` from the display’s data and
  **returns** it. Calling ``plot`` again does not mutate stored results or repeat heavy
  work unnecessarily; in Jupyter, a bare ``display.plot()`` at the end of a cell can
  show the figure, while in a script you typically assign the figure and call
  ``fig.show()`` if needed.
- ``set_style`` adjusts how subsequent ``plot`` calls render (for example line styles or
  colors), which helps when you want publication-ready output without rewriting the
  underlying computation.
- ``frame`` exposes the underlying data as a :class:`pandas.DataFrame`, so the same view
  you plotted is also available for filtering, exporting, or custom analysis.
- ``help`` lists what that display offers in your environment (for example via rich in
  the terminal or HTML in a notebook), which reduces guesswork when exploring a new
  chart type.

The example below follows that pattern: evaluate a model, ask the report for a ROC view,
then plot it.

.. plot::
    :context: close-figs
    :align: center

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from skore import evaluate

    X, y = make_classification(
        n_samples=10_000,
        n_classes=3,
        class_sep=0.3,
        n_clusters_per_class=1,
        random_state=42,
    )
    report = evaluate(LogisticRegression(), X, y, splitter=5)
    display = report.metrics.roc()
    display.plot()

Here :meth:`EstimatorReport.metrics.roc` returns a :class:`RocCurveDisplay`. Other
report facets expose other display types (confusion matrices, prediction errors, data
summaries, and so on); see the report sections in the reference and
:ref:`example_skore_api` for breadth.

The ``help`` method shows the available attributes and methods for a given display:

.. code-block:: python

    display.help()

You can restyle before plotting—for example the chance level on a ROC curve—then every
later ``plot`` uses those settings:

.. plot::
    :context: close-figs
    :align: center

    display.set_style(
        chance_level_kwargs=dict(linestyle="-", linewidth=5, color="tab:purple")
    )
    display.plot()

Finally, ``frame`` returns the tabular data backing the figure:

.. code-block:: python

    df = display.frame()
    df.head()

For the full list of display classes and report accessors, see the API reference.
