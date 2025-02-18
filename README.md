<div align="center">

  ![license](https://img.shields.io/pypi/l/skore)
  ![python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?style=flat&logo=python)
  [![downloads](https://static.pepy.tech/badge/skore/month)](https://pepy.tech/projects/skore)
  [![pypi](https://img.shields.io/pypi/v/skore)](https://pypi.org/project/skore/)
  [![Discord](https://img.shields.io/discord/1275821367324840119?label=Discord)](https://discord.probabl.ai/)

</div>

<div align="center">

  <picture>
    <source srcset="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Dark@2x.svg" media="(prefers-color-scheme: dark)">
    <img width="200" src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Light@2x.svg" alt="skore logo">
  </picture>
  <h3>the scikit-learn sidekick</h3>

Elevate ML Development with Built-in Recommended Practices \
[Documentation](https://skore.probabl.ai) — [Community](https://discord.probabl.ai)

</div>

<br />

## What is skore?

skore is a Python open-source library designed to help data scientists apply recommended practices and avoid common methodological pitfalls in scikit-learn.

## Key features

- **Diagnose**: catch methodological errors before they impact your models.
  - `train_test_split` supercharged with methodological guidance: the API is the same as scikit-learn's, but skore displays warnings when applicable. For example, it warns you against shuffling time series data or when you have class imbalance.
- **Evaluate**: automated insightful reports.
  - `EstimatorReport`: feed your scikit-learn compatible estimator and dataset, and it generates recommended metrics and plots to help you analyze your estimator. All these are computed and generated for you in 1 line of code. Under the hood, we use efficient caching to make the computations blazing fast.
  - `CrossValidationReport`: get a skore estimator report for each fold of your cross-validation.
  - `ComparisonReport`: benchmark your skore estimator reports.

## What's next?

Skore is just at the beginning of its journey, but we’re shipping fast! Frequent updates and new features are on the way as we work toward our vision of becoming a comprehensive library for data scientists.

⭐ Support us with a star and spread the word - it means a lot! ⭐

## 🚀 Quick start

### Installation

#### With pip

We recommend using a [virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html). You need `python>=3.9`.

Then, you can install skore by using `pip`:
```bash
pip install -U skore
```

#### With conda

skore is available in `conda-forge`:

```bash
conda install conda-forge::skore
```

You can find information on the latest version [here](https://anaconda.org/conda-forge/skore).

### Get assistance when developing your ML/DS projects

1. Evaluate your model using `skore.CrossValidationReport`:
    ```python
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    from skore import CrossValidationReport

    X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
    clf = LogisticRegression()

    cv_report = CrossValidationReport(clf, X, y)

    # Display the help tree to see all the insights that are available to you
    cv_report.help()
    ```

    ```python
    # Display the report metrics that was computed for you:
    df_cv_report_metrics = cv_report.metrics.report_metrics()
    df_cv_report_metrics
    ```

    ```python
    # Display the ROC curve that was generated for you:
    roc_plot = cv_report.metrics.roc()
    roc_plot.plot()
    ```

1. Store your results for safe-keeping.
    ```python
    # Create and load a skore project
    import skore
    my_project = skore.Project("my_project")
    ```

    ```python
    # Store your results
    my_project.put("df_cv_report_metrics", df_cv_report_metrics)
    my_project.put("roc_plot", roc_plot)
    ```

    ```python
    # Get your results
    df_get = my_project.get("df_cv_report_metrics")
    df_get
    ```

Learn more in our [documentation](https://skore.probabl.ai).


## Contributing

Thank you for considering contributing to skore! Join our mission to promote open-source and make machine learning development more robust and effective. Please check the contributing guidelines [here](https://github.com/probabl-ai/skore/blob/main/CONTRIBUTING.rst).


## Feedback & Community

-   Join our [Discord](https://discord.probabl.ai/) to share ideas or get support.
-   Request a feature or report a bug via [GitHub Issues](https://github.com/probabl-ai/skore/issues).

<br />

---

Brought to you by

<a href="https://probabl.ai" target="_blank">
    <picture>
        <source srcset="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Probabl-logo-orange.png" media="(prefers-color-scheme: dark)">
        <img width="120" src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Probabl-logo-blue.png" alt="Probabl logo">
    </picture>
</a>
