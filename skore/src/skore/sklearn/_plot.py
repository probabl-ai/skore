import pandas as pd
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.utils._optional_dependencies import check_matplotlib_support


def check_seaborn_support(caller_name):
    """Raise ImportError with detailed error message if seaborn is not installed.

    Plot utilities like any of the Display's plotting functions should lazily import
    matplotlib and call this helper before any computation.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires matplotlib.
    """
    try:
        import seaborn  # noqa
    except ImportError as e:
        raise ImportError(
            f"{caller_name} requires seaborn. You can install seaborn with "
            "`pip install seaborn`"
        ) from e


def _despine_matplotlib_axis(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_bounds(0, 1)


class RocCurveDisplay(RocCurveDisplay):
    def _plot_matplotlib(
        self,
        ax=None,
        *,
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        despine=False,
        **kwargs,
    ):
        super().plot(
            ax=ax,
            name=name,
            plot_chance_level=plot_chance_level,
            chance_level_kw=chance_level_kw,
        )
        if despine:
            _despine_matplotlib_axis(self.ax_)

        return self

    def _plot_plotly(
        self,
        ax=None,
        *,
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        despine=False,
        **kwargs,
    ):
        import plotly.graph_objects as go

        fig = go.Figure() if ax is None else ax

        # Add ROC curve trace
        name_str = ""
        if self.roc_auc is not None and name is not None:
            name_str = f"{name} (AUC = {self.roc_auc:0.2f})"
        elif self.roc_auc is not None:
            name_str = f"AUC = {self.roc_auc:0.2f}"
        elif name is not None:
            name_str = name

        fig.add_trace(
            go.Scatter(x=self.fpr, y=self.tpr, name=name_str, mode="lines", **kwargs)
        )

        if plot_chance_level:
            chance_level_kwargs = {
                "name": "Chance level (AUC = 0.5)",
                "line": {"color": "black", "dash": "dash"},
            }
            if chance_level_kw is not None:
                chance_level_kwargs.update(chance_level_kw)

            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", **chance_level_kwargs)
            )

        info_pos_label = (
            f" (Positive label: {self.pos_label})" if self.pos_label is not None else ""
        )

        fig.update_layout(
            xaxis_title=f"False Positive Rate{info_pos_label}",
            yaxis_title=f"True Positive Rate{info_pos_label}",
            xaxis=dict(range=[-0.01, 1.01]),
            yaxis=dict(range=[-0.01, 1.01]),
            showlegend=True,
            legend=dict(x=1, y=0, xanchor="right", yanchor="bottom"),
            width=600,
            height=600,
        )

        self.ax_ = fig
        self.figure_ = fig
        return self

    def plot(
        self,
        ax=None,
        *,
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        despine=False,
        backend="matplotlib",
        **kwargs,
    ):
        if backend == "matplotlib":
            return self._plot_matplotlib(
                ax=ax,
                name=name,
                plot_chance_level=plot_chance_level,
                chance_level_kw=chance_level_kw,
                despine=despine,
                **kwargs,
            )
        elif backend == "plotly":
            return self._plot_plotly(
                ax=ax,
                name=name,
                plot_chance_level=plot_chance_level,
                chance_level_kw=chance_level_kw,
                despine=despine,
                **kwargs,
            )
        else:
            raise ValueError(f"Backend '{backend}' is not supported.")


class WeightsDisplay:
    def __init__(self, weights, feature_names):
        self.weights = weights
        self.feature_names = feature_names

    def plot(
        self, ax=None, *, style="boxplot", add_data_points=False, backend="matplotlib"
    ):
        if backend == "matplotlib":
            return self._plot_matplotlib(
                ax=ax, style=style, add_data_points=add_data_points
            )
        elif backend == "plotly":
            return self._plot_plotly(
                ax=ax, style=style, add_data_points=add_data_points
            )
        else:
            raise ValueError(f"Backend '{backend}' is not supported.")

    def _plot_matplotlib(self, ax=None, *, style="boxplot", add_data_points=False):
        check_matplotlib_support(f"{self.__class__.__name__}.plot")
        check_seaborn_support(f"{self.__class__.__name__}.plot")
        import matplotlib.pyplot as plt
        import seaborn as sns

        if ax is None:
            _, ax = plt.subplots()

        df = pd.DataFrame(self.weights, columns=self.feature_names)
        if style == "boxplot":
            sns.boxplot(data=df, orient="h", color="tab:blue", whis=10, ax=ax)
        elif style == "violinplot":
            sns.violinplot(data=df, orient="h", color="tab:blue", ax=ax)
        else:
            raise ValueError(f"Style '{style}' is not supported.")

        if add_data_points:
            sns.stripplot(data=df, orient="h", palette="dark:k", alpha=0.8, ax=ax)

        ax.axvline(x=0, color=".5", linestyle="--")
        ax.set(xlabel="Feature weights", ylabel="Features")

        self.ax_ = ax
        self.figure_ = ax.figure
        return self

    def _plot_plotly(self, ax=None, *, style="boxplot", add_data_points=False):
        import plotly.graph_objects as go

        df = pd.DataFrame(self.weights, columns=self.feature_names)

        fig = go.Figure() if ax is None else ax

        # TODO: we could probably modify the dataframe to not have to loop here
        for col in df.columns[::-1]:  # reverse order to match matplotlib
            if style == "boxplot":
                fig.add_trace(
                    go.Box(
                        x=df[col],
                        name=col,
                        orientation="h",
                        whiskerwidth=1,
                        line_color="rgb(65, 105, 225)",
                        boxpoints="all" if add_data_points else False,
                        jitter=0.3 if add_data_points else 0,
                        pointpos=0 if add_data_points else None,
                        marker=dict(color="rgb(0,0,0)") if add_data_points else None,
                    )
                )
            elif style == "violinplot":
                fig.add_trace(
                    go.Violin(
                        x=df[col],
                        name=col,
                        orientation="h",
                        line_color="rgb(65, 105, 225)",
                        box_visible=True,
                        points="all" if add_data_points else False,
                        pointpos=0 if add_data_points else None,
                        marker=dict(color="rgb(0,0,0)") if add_data_points else None,
                    )
                )
            else:
                raise ValueError(f"Style '{style}' is not supported.")

        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            xaxis_title="Feature weights",
            yaxis_title="Features",
            showlegend=False,
        )

        self.ax_ = fig
        self.figure_ = fig
        return self

    @classmethod
    def from_cv_results(
        cls,
        cv_results,
        *,
        ax=None,
        style="boxplot",
        add_data_points=False,
        backend="matplotlib",
    ):
        estimators = cv_results["estimator"]
        if isinstance(estimators[0], Pipeline):
            estimators = [est.steps[-1][1] for est in estimators]
        weights = [est.coef_ for est in estimators]
        if hasattr(estimators[0], "feature_names_in_"):
            feature_names = estimators[0].feature_names_in_
        else:
            feature_names = [f"Feature #{i}" for i in range(len(weights[0]))]
        return cls(weights, feature_names).plot(
            ax=ax, style=style, add_data_points=add_data_points, backend=backend
        )
