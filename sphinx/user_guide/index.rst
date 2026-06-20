.. _user_guide:

User guide
==========

`skore` helps at structuring and storing what matters in your data science
experiments.

When it comes to data science, many libraries are available to help you investigate
your data. `pandas` or `polars` are great tools to explore and transform your data.
`skrub` is the one tool that brings the necessary *statefulness* to those
transformations required by a machine learning pipeline
(refer to `skrub's documentation <https://skrub-data.org/stable/documentation.html>`_).
`scikit-learn` and other `scikit-learn` compatible libraries (e.g. `xgboost`,
`lightgbm`) provide a set of algorithms to ingest transformed data and create
predictive models, as well as tools to diagnose and evaluate them.

All these libraries are broad and generic by design in order to accommodate a wide range
of use cases. It is your experience in choosing the appropriate building blocks from
those libraries that is the key to success.

`skore` is the package that ties all these pieces together. It allows you to
leverage your experience via a structured and robust framework
for understanding the impact of your analysis choices, with seamless integration
of all the above tools.

`skore` takes in the *full data science pipeline* created by assembling those
building blocks and provides **structured artifacts** that store the
information that matters for your particular use case. It minimizes unnecessary
overhead spent on busywork, such as navigating through documentation,
thereby streamlining the exploration process. `skore` reduces the amount of code
required to show the information that matters, removing boilerplate code and
making your project easier to understand and maintain. Finally, `skore` provides
a way to store these structured artifacts in a project, allowing you to
retrieve the results of your experiment whenever you need.

Table of contents
-----------------

.. toctree::
   :maxdepth: 2

   reporters
   automated_checks
   displays
   project
