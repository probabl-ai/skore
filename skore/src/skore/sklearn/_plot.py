from sklearn.metrics import RocCurveDisplay


def _despine_matplotlib_axis(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_bounds(0, 1)


class RocCurveDisplay(RocCurveDisplay):
    def plot(
        self,
        ax=None,
        *,
        name=None,
        plot_chance_level=False,
        chance_level_kw=None,
        despine=False,
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
