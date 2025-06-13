.. You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <div class="row row-padding-main-container">
      <div class="logo-container">
         <img src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Light@2x.svg" class="logo-landing only-light" alt="skore - Home"/>
         <img src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Dark@2x.svg" class="logo-landing only-dark pst-js-only" alt="skore - Home"/>
      </div>
      <h1 class="hero-title">Own Your Data Science</h1>
      <p class="hero-description">Elevate ML Development with Built-in Recommended Practices</p>
   </div>

🧩 What is Skore?
-----------------

**Skore** is a product whose core mission is to turn uneven ML development into structured, effective decision-making. It is made of two complementary components:

-  **Skore Lib**: the open-source Python library (described here!) designed to help data scientists boost their ML development with effective guidance and tooling.

-  **Skore Hub**: the collaborative layer where teams connect, learn more on our `product page <https://probabl.ai/skore>`_.

Key features of Skore Lib
^^^^^^^^^^^^^^^^^^^^^^^^^

**Evaluate and inspect**: automated insightful reports.

-  :class:`skore.EstimatorReport`: feed your scikit-learn compatible estimator and
   dataset, and it generates recommended metrics, feature importance, and plots to
   help you evaluate and inspect your model.
   All in just one line of code.
   Under the hood, we use efficient caching to make the computations blazing fast.

-  :class:`skore.CrossValidationReport`: get a skore estimator report for each fold
   of your cross-validation.

-  :class:`skore.ComparisonReport`: benchmark your skore estimator reports.

**Diagnose**: catch methodological errors before they impact your models.

-  :func:`skore.train_test_split` supercharged with methodological guidance:
   the API is the same as scikit-learn's, but skore displays warnings when
   applicable.
   For example, it warns you against shuffling time series data or when you have
   class imbalance.

🗓️ What's next?
---------------

Skore Lib is just at the beginning of its journey, but we’re shipping fast! Frequent updates and new features are on the way as we work toward our vision of becoming a comprehensive library for data scientists.

.. currentmodule:: skore

.. toctree::
   :maxdepth: 1
   :hidden:

   install
   user_guide/index
   auto_examples/index
   reference/index
   contributing

.. raw:: html

   <!-- Start of Reo Javascript -->
   <script type="text/javascript">
      !function(){var e,t,n;e="460c80a5e0b7c04",t=function(){Reo.init({clientID:"460c80a5e0b7c04"})},(n=document.createElement("script")).src="https://static.reo.dev/"+e+"/reo.js",n.defer=!0,n.onload=t,document.head.appendChild(n)}();
   </script>
   <!-- End of Reo Javascript -->
