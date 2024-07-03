# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

import altair as alt
import polars as pl
import numpy as np
from sktime.forecasting.compose import make_reduction

# +
X = pl.read_csv("X_train.csv").with_columns(
    pl.col("Time").str.to_datetime("%d/%m/%Y %H:%M")
)
y = pl.read_csv("Y_train.csv").join(X, on="ID").select("ID", "Time", "Production")

X_WF1 = X.filter(pl.col("WF") == "WF1")
y_WF1 = y.join(X_WF1, on="ID", how="semi")

from sktime.split import temporal_train_test_split

y_train, y_test, X_train, X_test = temporal_train_test_split(
    y=y_WF1["Production"].to_pandas(), X=X_WF1.drop(["ID", "WF", "Time"]).to_pandas()
)

fh = np.arange(1, len(y_test) + 1)  # forecasting horizon

# y_WF1_train = y_train.join(X_WF1_train, on="ID", how="semi")
# y_WF1_test = y_test.join(X_WF1_test, on="ID", how="semi")
# y_WF1_true = np.array(y_WF1_test["Production"])
# -


def cape(y_true, y_pred):
    """Compute the cumulated absolute percentage error."""
    return 100 * sum(abs(y_true - y_pred)) / sum(y_true)


capes = {}

# ## Baseline: Average production

y_pred = np.full_like(y_test, fill_value=y_train.mean())

capes["baseline"] = cape(y_test, y_pred)
cape(y_test, y_pred)

# ## HistGradientBoostingRegressor

# +
from sklearn.ensemble import HistGradientBoostingRegressor

regressor = HistGradientBoostingRegressor()
forecaster_hgbr = make_reduction(
    regressor,
    strategy="recursive",
    window_length=24,
)
# -

forecaster_hgbr.fit(fh=fh, y=y_train)

y_pred_hgbr = forecaster_hgbr.predict(fh=fh)

capes["hgbr"] = cape(y_test, y_pred_hgbr)
cape(y_test, y_pred_hgbr)

# ## Random Forest

# +
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()
forecaster_rf = make_reduction(
    regressor,
    strategy="recursive",
    window_length=12,
)

forecaster_rf.fit(y_train)
# -

y_pred_rf = forecaster_rf.predict(fh=fh)


capes["rf"] = cape(y_test, y_pred_rf)
cape(y_test, y_pred_rf)


# ## Use the exogenous variables

# +
regressor = RandomForestRegressor()
forecaster_exo = make_reduction(
    regressor,
    strategy="recursive",
    window_length=12,
)

forecaster_exo.fit(y_train)
# -

forecaster_exo.fit(y=y_train, X=X_train, fh=fh)

y_pred_exo = forecaster_exo.predict(fh=fh, X=X_test)


capes["rf_exo"] = cape(y_test, y_pred_exo)
cape(y_test, y_pred_exo)


alt.Chart(
    pl.DataFrame(capes)
    .transpose(include_header=True)
    .with_columns(
        pl.col("column").alias("experiment"), pl.col("column_0").alias("cape")
    )
).mark_bar().encode(x="experiment", y=alt.Y("cape").title("CAPE (%)")).properties(
    width=500
)
