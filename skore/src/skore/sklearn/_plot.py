import inspect

from sklearn.metrics import RocCurveDisplay


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
        # only in scikit-learn >= 1.6+
        parent_plot = super().plot
        plot_params = inspect.signature(parent_plot).parameters
        plot_kwargs = {
            "ax": ax,
            "name": name,
            "plot_chance_level": plot_chance_level,
            "chance_level_kw": chance_level_kw,
            **kwargs,
        }

        if "despine" in plot_params:
            plot_kwargs["despine"] = despine
            super().plot(**plot_kwargs)
        else:
            super().plot(**plot_kwargs)
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

        return self
