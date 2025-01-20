# %%
import os
import tempfile
from pathlib import Path

temp_dir = tempfile.TemporaryDirectory(prefix="skore_example_")
temp_dir_path = Path(temp_dir.name)

os.environ["POLARS_ALLOW_FORKING_THREAD"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# %%
import skore

project = skore.open(temp_dir_path / "my_project", overwrite=True)

# %%
from skrub.datasets import fetch_employee_salaries

datasets = fetch_employee_salaries()
df, y = datasets.X, datasets.y

# %%
from skrub import TableReport

table_report = TableReport(df)
table_report

# %%
project.put("df_report", table_report)

# %%
y

# %%
from skrub import TableVectorizer, TextEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    TableVectorizer(high_cardinality=TextEncoder()),
    HistGradientBoostingRegressor(),
)

# %%
from skore import CrossValidationReport

report = CrossValidationReport(model, X=df, y=y, cv_splitter=10, n_jobs=3)
report.help()

# %%
report.metrics.report_metrics()

# %%
report.metrics.report_metrics(aggregate=["mean", "std"])

# %%
