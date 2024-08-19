# ruff: noqa
import base64
from pathlib import Path
from time import time

import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from skrub import TableReport, tabular_learner
from skrub.datasets import fetch_employee_salaries
from tqdm import tqdm

from mandr import Mandr
from mandr.storage import FileSystem

DIR_MANDER = "datamander"
PATH_PROJECT = Path("skrub_demo")
N_SEEDS = 5


def init_ridge():
    return tabular_learner(RidgeCV())


def init_rf():
    return tabular_learner(RandomForestRegressor(n_jobs=4))


def init_gb():
    return tabular_learner(HistGradientBoostingRegressor())


INIT_MODEL_FUNC = {
    "ridge": init_ridge,
    "rf": init_rf,
    "gb": init_gb,
}


def evaluate_models(model_names):
    results = []
    for model_name in model_names:
        print(f"{' Evaluating ' + model_name + ' ':=^50}")
        results.append(evaluate_seeds(model_name))

    mander = get_mander(str(PATH_PROJECT))
    mander.insert("skrub_report", plot_skrub_report(), display_type="html")
    mander.insert("target_distribution", plot_y_distribution())
    mander.insert("Metrics", plot_table_metrics(results))
    mander.insert("R2 vs fit time", plot_r2_vs_fit_time(results), display_type="html")


def evaluate_seeds(model_name):
    path_model = PATH_PROJECT / model_name

    seed_scores = []
    for random_state in tqdm(range(N_SEEDS)):
        bunch = get_data(random_state)
        model = INIT_MODEL_FUNC[model_name]()

        tic = time()
        model.fit(bunch.X_train, bunch.y_train)
        fit_time = time() - tic

        scores = evaluate(model, bunch)
        scores.update(
            {
                "random_state": random_state,
                "model_name": model_name,
                "fit_time": fit_time,
            }
        )

        path_seed = path_model / f"random_state{random_state}"
        mander = get_mander(str(path_seed))
        mander.insert("scores", scores)  # scores is a dict
        mander.insert("model_repr", plot_model_repr(model), display_type="html")
        mander.insert("feature importance", plot_feature_importance(model, bunch))
        seed_scores.append(scores)

    agg_scores = aggregate_seeds_results(seed_scores)
    mander = get_mander(str(path_model))
    mander.insert("agg_scores", agg_scores)

    return agg_scores


def evaluate(model, bunch):
    y_pred = model.predict(bunch.X_test)
    y_test = bunch["y_test"]

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    scores = {
        "y_pred": y_pred.tolist(),
        "r2": r2,
        "mae": mae,
        "mse": mse,
    }

    return scores


def aggregate_seeds_results(scores):
    agg_score = dict()
    for metric in ["r2", "mae", "mse", "fit_time"]:
        score_seeds = [score[metric] for score in scores]
        agg_score.update(
            {
                f"mean_{metric}": np.mean(score_seeds),
                f"std_{metric}": np.std(score_seeds),
            }
        )

    agg_score["model_name"] = scores[0]["model_name"]

    return agg_score


def get_data(random_state, split=True):
    dataset = fetch_employee_salaries()
    X, y = dataset.X, dataset.y
    if split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state
        )
        return Bunch(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    else:
        return Bunch(X=X, y=y)


def plot_table_metrics(results):
    df = pd.DataFrame(results)
    rename = {
        "r2": "R2 (↑)",
        "mse": "MSE (↓)",
        "mae": "MAE (↓)",
        "fit_time": "Fit time (↓)",
    }

    for metric in ["r2", "mae", "mse", "fit_time"]:
        mean_key, std_key = f"mean_{metric}", f"std_{metric}"
        df[rename[metric]] = (
            df[mean_key].round(4).astype(str) + " ± " + df[std_key].round(4).astype(str)
        )
        df = df.drop([mean_key, std_key], axis=1)

    return df


def plot_r2_vs_fit_time(results):
    df = pd.DataFrame(results)

    model_names = df["model_name"].tolist()
    palette = dict(
        zip(
            list(model_names),
            sns.color_palette("colorblind", n_colors=len(model_names)),
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    c = "black"
    plt.errorbar(
        x=df["mean_fit_time"],
        y=df["mean_r2"],
        yerr=df["std_r2"],
        fmt="none",
        c=c,
        capsize=2,
    )
    plt.errorbar(
        x=df["mean_fit_time"],
        xerr=df["std_fit_time"],
        y=df["mean_r2"],
        fmt="none",
        c=c,
        capsize=2,
    )
    ax = sns.scatterplot(
        df,
        x="mean_fit_time",
        y="mean_r2",
        hue="model_name",
        s=200,
        palette=palette,
        zorder=10,
        alpha=1,
    )

    ax.grid()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # plt.tight_layout()

    filename = "r2_vs_fit_time.png"
    plt.savefig(filename)

    return convert_fig_to_html(filename)


def plot_skrub_report():
    bunch = get_data(random_state=0, split=False)
    df = pd.concat([bunch.X, bunch.y], axis=1)
    return TableReport(df).html()


def plot_feature_importance(model, bunch):
    importances = permutation_importance(model, bunch.X_test, bunch.y_test, n_jobs=4)

    feature_imp = pd.DataFrame(
        importances["importances"].T, columns=bunch.X_train.columns
    ).melt()  # Convert the dataframe to a long format

    fig = (
        alt.Chart(feature_imp)
        .mark_boxplot(extent="min-max")
        .encode(
            alt.X("value:Q").scale(domain=[0, 1]),
            alt.Y("variable:N"),
        )
    )

    return fig


def plot_y_distribution():
    bunch = get_data(random_state=0, split=False)
    df = pd.concat([bunch.X, bunch.y], axis=1)
    N = min(1000, df.shape[0])
    df = df.sample(N)

    alt.data_transformers.enable("vegafusion")
    fig = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("current_annual_salary:Q", bin=alt.Bin(maxbins=30)),
            y="count()",
            color="gender:N",
        )
        .properties(width=600, height=400)
        .interactive()
    )

    return fig


def plot_model_repr(model):
    return model._repr_html_()


def get_mander(path):
    return Mandr(
        path,
        storage=FileSystem(directory=DIR_MANDER),
    )


def delete_mander(mander):
    for k in list(mander):
        mander.delete(k)


def convert_fig_to_html(path_file):
    data_uri = base64.b64encode(open(path_file, "rb").read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{data_uri}">'


if __name__ == "__main__":
    evaluate_models(model_names=list(INIT_MODEL_FUNC))
