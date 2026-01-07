<div align="center">

  ![license](https://img.shields.io/pypi/l/skore)
  ![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=flat&logo=python)
  [![downloads](https://static.pepy.tech/badge/skore/month)](https://pepy.tech/projects/skore)
  [![pypi](https://img.shields.io/pypi/v/skore)](https://pypi.org/project/skore/)
  [![Discord](https://img.shields.io/discord/1275821367324840119?label=Discord)](https://discord.probabl.ai/)

</div>

<div align="center">

  <picture>
    <source srcset="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Dark@2x.svg" media="(prefers-color-scheme: dark)">
    <img width="200" src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Logo_Skore_Light@2x.svg" alt="skore logo">
  </picture>
  <h3>Own Your Data Science</h3>

Elevate ML Development with Built-in Recommended Practices \
[Documentation](https://docs.skore.probabl.ai) — [Community](https://discord.probabl.ai) — [YouTube](https://youtube.com/playlist?list=PLSIzlWDI17bTpixfFkooxLpbz4DNQcam3) — [Skore Hub](https://probabl.ai/skore)

</div>

<br />

## 🎯 Why Skore?

When it comes to data science, you have excellent tools at your disposal: `pandas` and `polars` for data exploration, `skrub` for stateful transformations, and `scikit-learn` for model training and evaluation. These libraries are designed to be generic and accommodate a wide range of use cases.

**But here's the challenge**: Your experience is key to choosing the right building blocks and methodologies. You often spend significant time navigating documentation, writing boilerplate code for common evaluations, and struggling to maintain clear project structure.

**Skore is the conductor** that transforms your data science pipeline into structured, meaningful artifacts. It reduces the time you spend on documentation navigation, eliminates boilerplate code, and guides you toward the right methodological information to answer your questions.

### What Skore does for you:

- **Structures your experiments**: Automatically generates the insights that matter for your use case
- **Reduces boilerplate**: One line of code gives you comprehensive model evaluation
- **Guides your decisions**: Built-in methodological warnings help you avoid common pitfalls
- **Maintains clarity**: Structured project organization makes your work easier to understand and maintain

⭐ Support us with a star and spread the word - it means a lot! ⭐

## 🧩 What is Skore?

The core mission of **Skore** is to turn uneven ML development into structured, effective decision-making. It consists of two complementary components:
- **Skore Lib**: the open-source Python library (described here!) that provides the structured artifacts and methodological guidance for your data science experiments.
- **Skore Hub**: the collaborative platform where teams can share, compare, and build upon each other's structured experiments. Learn more on our [product page](https://probabl.ai/skore).

## ⚡️ Quick start

### Installation

#### With pip

We recommend using a [virtual environment (venv)](https://docs.python.org/3/tutorial/venv.html). You need `python>=3.10`.

Then, you can install skore by using `pip`:
```bash
# If you plan to use Skore locally
pip install -U skore
# If you wish to interact with Skore Hub as well
pip install -U skore[hub]
```

#### With conda

skore is available in `conda-forge` both for local and hub use:

```bash
conda install conda-forge::skore
```

You can find information on the latest version [here](https://anaconda.org/conda-forge/skore).

### Get structured insights from your ML pipeline

Evaluate your model and get comprehensive insights in one line:

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from skore import CrossValidationReport

X, y = make_classification(n_classes=2, n_samples=100_000, n_informative=4)
clf = LogisticRegression()

# Get structured insights that matter for your use case
cv_report = CrossValidationReport(clf, X, y)

# See what insights are available
cv_report.help()

# Example: Access the metrics summary
metrics_summary = cv_report.metrics.summarize().frame()

# Example: Get the ROC curve
roc_plot = cv_report.metrics.roc()
roc_plot.plot()
```

Learn more in our [documentation](https://docs.skore.probabl.ai).

## 🛠️ Contributing

Join our mission to promote open-source and make machine learning development more robust and effective. If you'd like to contribute, please check the contributing guidelines [here](https://github.com/probabl-ai/skore/blob/main/CONTRIBUTING.rst).

## 👋 Feedback & Community

-   Join our [Discord](https://discord.probabl.ai/) to share ideas or get support.
-   Request a feature or report a bug via [GitHub Issues](https://github.com/probabl-ai/skore/issues).

## Support

Skore is tested on Linux and Windows, for at most 4 versions of Python, and at most 4 versions of scikit-learn:
- Python 3.11
  - scikit-learn 1.5
  - scikit-learn 1.8
- Python 3.12
  - scikit-learn 1.5
  - scikit-learn 1.8
- Python 3.13
  - scikit-learn 1.5
  - scikit-learn 1.8
- Python 3.14
  - scikit-learn 1.5
  - scikit-learn 1.6
  - scikit-learn 1.7
  - scikit-learn 1.8

---

Brought to you by

<a href="https://probabl.ai/skore" target="_blank">
    <picture>
        <source srcset="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Probabl-logo-orange.png" media="(prefers-color-scheme: dark)">
        <img width="120" src="https://media.githubusercontent.com/media/probabl-ai/skore/main/sphinx/_static/images/Probabl-logo-blue.png" alt="Probabl logo">
    </picture>
</a>
