<div align="center">

  <picture>
    <source srcset="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Dark@2x.svg" media="(prefers-color-scheme: dark)">
    <img width="200" src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Light@2x.svg" alt="skore logo">
  </picture>
  <h3>the scikit-learn sidekick</h3>

Elevate ML Development with Tracking and Built-in Recommended Practices \
[Documentation](https://skore.probabl.ai) — [Community](https://discord.probabl.ai)

</div>

<br />

## Why skore?

ML development is an art — blending business sense, stats knowledge, and coding skills. Brought to you by [Probabl](https://probabl.ai), a company co-founded by scikit-learn core developers, skore helps data scientists focus on what matters: building impactful models with hindsight and clarity.

Skore is just at the beginning of its journey, but we’re shipping fast! Frequent updates and new features are on the way as we work toward our vision of becoming a comprehensive library for data scientists, supporting every phase of the machine learning lifecycle.

⭐ Support us with a star and spread the word - it means a lot! ⭐


## Key features

- **Track and Visualize Results**: Capture your intermediate ML/DS results without the overhead, while gaining deeper insights through intuitive visualizations of your experiments.
- **Elevate Model Development**: Avoid common pitfalls and follow recommended practices with automatic guidance and insights.
    - Enhancing key scikit-learn features with `skore.CrossValidationReporter` and `skore.train_test_split()`.

![GIF: short demo of skore](https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/2024_12_12_skore_demo_comp.gif)

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

1. From your Python code, create and load a skore project:
    ```python
    import skore
    my_project = skore.open("my_project")
    ```
    This will create a skore project directory named `my_project.skore` in your current working directory.

2. Evaluate your model using `skore.CrossValidationReporter`:
    ```python
    from sklearn.datasets import load_iris
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X, y = load_iris(return_X_y=True)
    clf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression())
    ])

    reporter = skore.CrossValidationReporter(clf_pipeline, X, y, cv=5)

    # Store the results in the project
    my_project.put("cv_reporter", reporter)

    # Display a plot result in your notebook
    reporter.plots.scores
    ```

3. Finally, from your shell (in the same directory), start the UI:
    ```bash
    skore launch "my_project"
    ```
    This will open skore-ui in a browser window.

You will automatically be able to visualize some key metrics (although you might have forgotten to specify all of them):
![Cross-validation screenshot](https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/2024_12_12_skore_cross_val_clf.png)

Also check out `skore.train_test_split()` that enhances scikit-learn. Learn more in our [documentation](https://skore.probabl.ai).


## Contributing

Thank you for considering contributing to skore! Join our mission to promote open-source and make machine learning development more robust and effective. Please check the contributing guidelines [here](https://github.com/probabl-ai/skore/blob/main/CONTRIBUTING.rst).


## Feedback & Community

-   Join our [Discord](https://discord.probabl.ai/) to share ideas or get support.
-   Request a feature or report a bug via [GitHub Issues](https://github.com/probabl-ai/skore/issues).

<br />

<div align="center">

  ![license](https://img.shields.io/pypi/l/skore)
  ![python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?style=flat&logo=python)
  [![downloads](https://static.pepy.tech/badge/skore/month)](https://pepy.tech/projects/skore)
  [![pypi](https://img.shields.io/pypi/v/skore)](https://pypi.org/project/skore/)
  [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.probabl.ai/)

</div>
---

Brought to you by

<a href="https://probabl.ai" target="_blank">
    <picture>
        <source srcset="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Probabl-logo-orange.png" media="(prefers-color-scheme: dark)">
        <img width="120" src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Probabl-logo-blue.png" alt="Probabl logo">
    </picture>
</a>
