.. _getting_started:

Getting started
===============

.. currentmodule:: skore

From your shell, initialize a skore project, here named ``my_project``:

.. code-block:: bash

    python -m skore create "my_project"

This will create a skore project directory named ``my_project.skore`` in your
current working directory.

Now that the project file exists, we can write some Python code to put some 
useful things in it. 

.. code-block:: python

    from sklearn.datasets import make_regression
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Ridge
    import numpy as np
    import polars as pl 
    import altair as alt
    from skore import load

    # Search for a hyperparameter alpha on a simulated dataset
    X, y = make_regression(n_samples=1000, noise=100)

    cv = GridSearchCV(
        Ridge(), 
        param_grid={"alpha": np.logspace(-3, 5, 100)},
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )
    cv.fit(X, y)

    # Store the hyper parameters metrics in a dataframe and make a custom altair chart.
    df = pl.DataFrame(cv.cv_results_).with_columns(mse=-pl.col("mean_test_score"))

    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("param_alpha").scale(type="log"), 
        y='mse'
    ).properties(
        title="mean squared error vs. alpha"
    )

    # Store relevant information into a skore project. 
    skore_proj = load("my_project")
    skore_proj.put("cv", cv)
    skore_proj.put("df", df)
    skore_proj.put("chart", chart)


Now that we have some elements in our project, we can explore them in a web interface by running:

.. code-block:: bash

    python -m skore launch "my_project"

This will automatically open a browser and allow you to click and drag elements into a view which you can then easily share statically with stakeholders. To do this you'll first want to create a view by clicking the ``+`` in the upper left side. From there you can add elemensts as you see fit.

The goal of `skore` is to make it easy to share results of your experiments and you can also imagine that we will add useful widgets to help communicate results during the entire data science lifecycle. 

For more features, please look into :ref:`auto_examples`.

.. image:: https://raw.githubusercontent.com/sylvaincom/sylvaincom.github.io/master/files/probabl/skore/2024_10_14_skore_demo.gif
   :alt: Getting started with ``skore`` demo