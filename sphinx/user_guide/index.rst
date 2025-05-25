.. _user_guide:

User guide
==========

`skore` helps at structuring and storing what matters in your data science
experiments.

When it comes to data science, many libraries are available to help you experiment.
`pandas` or `polars` are great tools to explore and transform your data. `skrub` is the
one tool that brings the necessary *statefullness* to those transformations required
by the machine learning pipeline. `scikit-learn` and other `scikit-learn` compatible
libraries (e.g. `xgboost`, `lightgbm`) provide a set of algorithms to ingest those
transformed data and create predictive models. `scikit-learn` provides even more tools
to diagnose and evaluate those models.

`skore` is the cherry on the top. All those libraries are thought to be generic to
accommodate a wide range of use cases. When it comes to your particular use case,
your experience is the key to success by choosing the appropriate building blocks from
those libraries.

`skore` intends to *consume* the data science pipeline created by assembling those
libraries components and provide **structured artifacts** that would store the
information that matters for your use case. It will reduce the amount of time to
navigate through the documentation and guide you towards the right methodological
information to answer your questions. `skore` will also reduce the amount of code
required to show the information that matters, removing boilerplate code, making your
project easier to understand and maintain in the long run. Finally, `skore` provides
a way to store all those structured artifacts in a structured project and thus help
you later on to retrieve the experiment results that you need.

Table of contents
-----------------

.. toctree::
   :maxdepth: 2

   reporters
   project
