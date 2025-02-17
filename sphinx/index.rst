.. You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <div class="row row-padding-main-container">
      <div class="logo-container">
         <img src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Light@2x.svg" class="logo-landing only-light" alt="skore - Home"/>
         <img src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Dark@2x.svg" class="logo-landing only-dark pst-js-only" alt="skore - Home"/>
      </div>
      <h1 class="hero-title">the scikit-learn sidekick</h1>
      <p class="hero-description">Elevate ML Development with Built-in Recommended Practices</p>
   </div>

.. admonition:: Where to start?

   See our :ref:`example_quick_start` page!

What is skore?
""""""""""""""

skore is a Python open-source library designed to help data scientists apply recommended
practices and avoid common methodological pitfalls in scikit-learn.

Key features
""""""""""""

-  **Diagnose**: catch methodological errors before they impact your models.

   -  :func:`skore.train_test_split` supercharged with methodological guidance:
      the API is the same as scikit-learn's, but skore displays warnings when
      applicable.
      For example, it warns you against shuffling time series data or when you have
      class imbalance.

-  **Evaluate**: automated insightful reports.

   -  :class:`skore.EstimatorReport`: feed your scikit-learn compatible estimator and
      dataset, and it generates recommended metrics and plots to help you analyze your
      estimator.
      All these are computed and generated for you in 1 line of code.
      Under the hood, we use efficient caching to make the computations blazing fast.

   -  :class:`skore.CrossValidationReport`: get a skore estimator report for each fold
      of your cross-validation.

   -  :class:`skore.ComparisonReport`: benchmark your skore estimator reports.

What's next?
""""""""""""

Skore is just at the beginning of its journey, but we’re shipping fast! Frequent updates and new features are on the way as we work toward our vision of becoming a comprehensive library for data scientists.


.. currentmodule:: skore

.. toctree::
   :maxdepth: 1
   :hidden:

   install
   auto_examples/index
   reference/index
   contributing
