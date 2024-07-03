# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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

# +
import altair as alt
import polars as pl
import polars.selectors as cs

# Needed for larger datasets
alt.data_transformers.enable("vegafusion")

# alt.renderers.enable("browser")
# -

X = pl.read_csv("X_train.csv")
y = pl.read_csv("Y_train.csv")
X = X.with_columns(pl.col("Time").str.to_datetime("%d/%m/%Y %H:%M"))
X_y = X.join(y, on="ID")
# For now, let us focus on just one of the 6 wind farms so that we understand the structure of the data.

X_WF1 = X.filter(pl.col("WF") == "WF1")
X_y_WF1 = X_y.filter(pl.col("WF") == "WF1")
X_WF1.filter(pl.col("Time").dt.hour() == 0)
X_WF1["Time"].max()
# Let's look at column `NWP1_00h_D-2_U`.
#
# According to the docs, this is what this column name means:
# "The zonal component of wind speed, $U$, right now, as predicted two days ago (`D-2`), at midnight (`00h`), by weather station no. 1 (`NWP1`)"

alt.Chart(X_WF1).mark_line().encode(
    x="Time:T", y="NWP1_00h_D-2_U:Q", tooltip=alt.Tooltip("Time:T", format="%H:%M")
).interactive()
# The weather prediction columns have many null values. Is there a pattern in those?

(
    list(
        X_WF1.filter(pl.col("NWP1_00h_D-2_U").is_not_null())["Time"].dt.hour().unique()
    ),
    list(
        X_WF1.filter(pl.col("NWP1_06h_D-2_U").is_not_null())["Time"].dt.hour().unique()
    ),
    list(
        X_WF1.filter(pl.col("NWP1_12h_D-2_U").is_not_null())["Time"].dt.hour().unique()
    ),
    list(
        X_WF1.filter(pl.col("NWP1_18h_D-2_U").is_not_null())["Time"].dt.hour().unique()
    ),
)
# We have some complementary data about the wind farms. Let's take a short look.

# +
X_comp = pl.read_csv("WindFarms_complementary_data.csv", separator=";")
X_comp = X_comp.filter(pl.col("Time (UTC)").is_not_null())
X_comp = X_comp.with_columns(pl.col("Time (UTC)").str.to_datetime("%d/%m/%Y %H:%M"))

(
    alt.Chart(
        X_comp.filter(
            (pl.col("Wind Farm") == "WF1") & (pl.col("Wind Turbine") == "TE1")
        ).with_columns(
            (pl.col("Wind direction (�)") - pl.col("Nacelle direction (�)")).alias(
                "Nacelle misalignment (deg)"
            )
        )
    )
    .mark_point()
    .encode(x="Time (UTC)", y="Nacelle misalignment (deg)")
)
# -

# Now let us look at the target column specifically.

# Statistics of power production
X_y_WF1["Production"].describe()


# +
# Histogram
(
    alt.Chart(X_y_WF1)
    .mark_bar()
    .encode(
        alt.X("Production", bin=alt.Bin(step=0.5)).title("Production (MWh)"),
        y=alt.Y("count()").title("Frequency"),
    )
    .properties(width=800)
)

# The distribution of production is heavily right skewed. The median is 0.82 MW.
# According to Our World in Data 2017 (https://ourworldindata.org/scale-for-electricity), a French person consumes 0.019 MWh/day
# -

# Total production for the month
(
    alt.Chart(X_y_WF1)
    .mark_line()
    .encode(
        x="yearmonth(Time):T",
        y=alt.Y("sum(Production)").title("Total production (MWh)"),
    )
    .properties(width=800)
)

# There's a big drop in December 2018, compared to November and January. Is it because demand dropped, or because the data was corrupted, or because the wind farms were in maintenance?
#

(
    alt.Chart(
        X_y_WF1.filter(pl.col("Time").dt.month() == 12),
        title="Hourly production in December 2018",
    )
    .mark_line()
    .encode(x="Time", y="Production")
    .properties(width=1000)
)

# It turns out the power production was very near zero for 9 consecutive days, from 12 December to 21 December.
#

# Now let us try training a model using the classic `sktime` workflow, to get become familiar with it.

# +
import numpy as np
from sktime.forecasting.compose import make_reduction
from sklearn.ensemble import RandomForestRegressor
from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
from sktime.split import temporal_train_test_split

# Format the data to be sktime-friendly
y_train, y_test, X_train, X_test = temporal_train_test_split(
    y=X_y_WF1["Production"].to_pandas(), X=X_WF1.drop(["ID", "WF", "Time"]).to_pandas()
)

fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
regressor = RandomForestRegressor()
forecaster = make_reduction(
    regressor,
    strategy="recursive",
    window_length=12,
)
# -

# Takes a while
forecaster.fit(y=y_train, X=X_train, fh=fh)

y_pred = forecaster.predict(fh=fh, X=X_test)
smape = MeanAbsolutePercentageError()
smape(y_test, y_pred)

# Show predictions with test
df = (
    pl.DataFrame({"y_pred": y_pred, "y_test": y_test})
    .with_row_index()
    .unpivot(index="index")
)
alt.Chart(
    df, title="Comparison of predicted and true production on the test set"
).mark_line().encode(
    x="index", y=alt.X("value").title("Production (MWh)"), color="variable"
)
# It's not great

# Let us explore the data some more to find patterns in the target.

# +
# Average production depending on the day of the week
(
    alt.Chart(X_y_WF1.with_columns(pl.col("Time").dt.weekday().alias("Day of week")))
    .mark_bar()
    .encode(x="Day of week", y="mean(Production)")
    + alt.Chart(X_y_WF1.with_columns(pl.col("Time").dt.weekday().alias("Day of week")))
    .mark_errorbar(extent="iqr")
    .encode(x="Day of week", y="Production")
)
# 1 is Monday, 7 is Sunday
# Top production is on Mondays and Sundays, bottom is Thursdays

# Error bars are the IQR
# +
# Average production depending on month of the year
base = alt.Chart(X_y_WF1.with_columns(pl.col("Time").dt.month().alias("Month")))
(
    base.mark_bar().encode(x="Month", y="mean(Production)")
    + base.mark_errorbar(extent="iqr").encode(x="Month", y="Production")
)

# 1 is January, 12 is December
# Top production is January by far, bottom is August/September
# December is low, as mentioned earlier

# The error bars show the inter-quartile range (bottom is 25% quantile, top is 75% quantile)
# This way we can clearly see that a lot of the data is very close to 0
# -
# Let us see if the different predictions of the same meteorological variable ($U$) are coherent with each other. If so, we should remove some of them or combine them.

import polars.selectors as cs

nwp1 = X_y_WF1.select(cs.matches("Time") | (cs.matches("NWP1") & cs.matches("_U")))
alt.Chart(
    nwp1.unpivot(index="Time"), title="Predicted value of U from different forecasts"
).mark_point().encode(
    x="Time",
    y=alt.Y("value").title("Predicted value of U (m/s)"),
    color=alt.Color(
        "variable", legend=alt.Legend(orient="left", title="Prediction time")
    ),
).properties(width=5000, height=500)
X_y_WF1.with_columns(
    mean_U=pl.mean_horizontal((cs.matches("NWP1") & cs.matches("_U"))),
    min_U=pl.min_horizontal((cs.matches("NWP1") & cs.matches("_U"))),
    max_U=pl.max_horizontal((cs.matches("NWP1") & cs.matches("_U"))),
)

(
    alt.Chart(nwp1.melt(id_vars="Time")).mark_line().encode(x="Time", y="mean(value)")
    + alt.Chart(nwp1.melt(id_vars="Time"))
    .mark_errorband(extent="ci")
    .encode(x="Time", y="value")
).properties(width=5000, height=500)

# Let us further study patterns in the target variable, to see if it could be sufficient to build a model just based on past values (and not any of the other variables).

# +
# Auto-correlation of y

from sktime.transformations.series.acf import AutoCorrelationTransformer

transformer = AutoCorrelationTransformer()
y_hat = transformer.fit_transform(X_y_WF1["Production"].to_pandas())
# -

alt.Chart(pl.DataFrame(dict(y_hat=y_hat)).with_row_index()).mark_point().encode(
    x="index", y="y_hat"
)

from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(X_y_WF1["Production"], lags=50)

# Let's try to build another forecaster based on a different sklearn regressor.

# +
from sklearn.ensemble import HistGradientBoostingRegressor
from sktime.split import temporal_train_test_split
import numpy as np
from sktime.forecasting.compose import make_reduction

y_train, y_test, X_train, X_test = temporal_train_test_split(
    y=X_y_WF1["Production"].to_pandas(), X=X_WF1.drop(["ID", "WF", "Time"]).to_pandas()
)

fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
regressor = HistGradientBoostingRegressor()
forecaster = make_reduction(
    regressor,
    strategy="recursive",
    window_length=24,
)
# -

forecaster.fit(fh=fh, y=y_train)

y_pred = forecaster.predict(fh=fh)
y_pred

df = pl.DataFrame({"y_pred": y_pred, "y_test": y_test}).with_row_index().melt("index")
alt.Chart(
    df, title="Comparison of predicted and true production on the test set"
).mark_line().encode(
    x="index", y=alt.X("value").title("Production (MWh)"), color="variable"
)
# Terrible

# Let us look at the correlations between each exogenous variable and the energy production.

alt.Chart(X_y_WF1).mark_point().encode(
    x=alt.X(alt.repeat("column"), type="quantitative"), y="Production"
).repeat(column=X_y_WF1.columns[3:-1])

# According to the docs it might be useful to compute the actual wind speed from the $U$ and $V$ components. Let's take a look!

import itertools

# +
speed_columns = [
    (
        pl.col(f"{station}_{hour}_{day}_U") ** 2
        + pl.col(f"{station}_{hour}_{day}_V") ** 2
    )
    .sqrt()
    .alias(f"{station}_{hour}_{day}_Speed")
    for station, hour, day in itertools.product(
        ["NWP1", "NWP2", "NWP3", "NWP4"],
        ["00h", "06h", "12h", "18h"],
        ["D-2", "D-1", "D"],
    )
    if f"{station}_{hour}_{day}_U" in X_y_WF1.columns
]

X_y_WF1.with_columns(speed_columns)
# -

with pl.Config(tbl_rows=-1):
    print(
        X_y_WF1.with_columns(speed_columns)
        .select(
            [
                pl.corr("Production", a).alias(f"corr_{a}")
                for a in X_y_WF1.with_columns(speed_columns).columns
            ]
        )
        .transpose(include_header=True)
    )

alt.Chart(X_y_WF1.with_columns(speed_columns)).mark_point().encode(
    x=alt.X(alt.repeat("column"), type="quantitative"), y="Production"
).repeat(
    column=[
        col for col in X_y_WF1.with_columns(speed_columns).columns if "Speed" in col
    ]
)


# "Total" Wind speed is indeed highly correlated with electricity production. One question these plots prompt is:
# How often do we see high wind speeds, yet production at or near 0?

with pl.Config(tbl_rows=-1):
    print(
        X_y_WF1.with_columns(speed_columns)
        .select(
            [
                pl.corr("Production", a).alias(f"corr_{a}")
                for a in X_y_WF1.with_columns(speed_columns).columns
                if "Speed" in a
            ]
        )
        .transpose(include_header=True)
    )


with pl.Config(tbl_rows=-1):
    print(
        X_y_WF1.with_columns(speed_columns)
        .filter(pl.col("Production") > 0.05)
        .select(
            [
                pl.corr("Production", a).alias(f"corr_{a}")
                for a in X_y_WF1.with_columns(speed_columns).columns
                if "Speed" in a
            ]
        )
        .transpose(include_header=True)
    )

# When you remove rows with small elecricity production, the correlation between speed and production decreases!
# This might indicate that there are many occurrences with very low wind speeds and very low production.
#
# I'm curious: When the production is very low, how often is it that the wind speed is low?

(
    alt.Chart(X_y_WF1.with_columns(speed_columns).filter(pl.col("Production") < 0.05))
    .mark_bar()
    .encode(
        x=alt.X("NWP1_00h_D-2_Speed", bin=True, title="NWP1_00h_D-2_Speed (m/s)"),
        y=alt.Y("count()", title="Frequency"),
    )
)

# At a glance it looks abnormal -- we'd expect very low speeds to be more frequent. However, we need to get an idea of scales here: according to the [Beaufort wind force scale](https://en.wikipedia.org/wiki/Beaufort_scale), 5 m/s corresponds to a "Gentle Breeze". Thinking about it more closely, it makes sense that, at 100m altitude, the wind always blows at least a little.

# Let us try yet another sklearn regressor.

# +
fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
regressor = RandomForestRegressor()
forecaster = make_reduction(
    regressor,
    strategy="recursive",
    window_length=12,
)

forecaster.fit(y_train)
# -

df = (
    pl.DataFrame({"y_pred": forecaster.predict(fh), "y_test": y_test})
    .with_row_index()
    .melt("index")
)
(
    alt.Chart(df).mark_line().encode(x="index", y="value", color="variable")
    + alt.Chart().mark_rule().encode(y=alt.datum(y_test.mean()))
)


# It appears the model is predicting a constant close to the mean of `y_test`.
