from typing import Any, Literal, ClassVar

from .media import Media, Representation


EstimatorReport = Any


class FeatureImportance(Media):
    report: EstimatorReport = Field(repr=False, exclude=True)
    accessor: ClassVar[str]
    category: Literal["feature_importance"] = "feature_importance"

    @computed_field
    @cached_property
    def representation(self) -> Representation:
        try:
            function = reduce(getattr, self.accessor.split("."), self.report)
        except AttributeError:
            return None

        return Representation(
            media_type="application/vnd.dataframe",
            value=self.__raw__.fillna("NaN").to_dict(orient="tight"),
        )


        table_report_display = self.report.data.analyze(data_source=self.data_source)

        return Representation(
            media_type="application/vnd.skrub.table-report.v1+html",
            value=table_report_display._html_repr(),
        )


class TableReportTrain(Media):
    data_source: ClassVar[Literal["train"]]: "train"
    attributes: Literal[{"data_source": "train"}] = {"data_source": "train"}


class TableReportTest(Media):
    data_source: ClassVar[Literal["test"]]: "test"
    attributes: Literal[{"data_source": "test"}] = {"data_source": "test"}


def pd(self, name, verbose_name, category, **kwargs) -> Representation | None:
    """Return sub-representation made of ``pandas`` dataframes."""
    try:
        function = attrgetter(name)(self.report)
    except AttributeError:
        return None
    else:
        function_parameters = signature(function).parameters
        function_kwargs = {k: v for k, v in kwargs.items() if k in function_parameters}
        dataframe = function(**function_kwargs)
        item = PandasDataFrameItem.factory(dataframe)

        return {
            "key": name.split(".")[-1],
            "verbose_name": verbose_name,
            "category": category,
            "attributes": kwargs,
            "parameters": {},
            "representation": item.__representation__["representation"],
        }


self.pd(
    "feature_importance.permutation",
    "Feature importance - Permutation",
    "feature_importance",
    data_source="train",
    method="permutation",
),
self.pd(
    "feature_importance.permutation",
    "Feature importance - Permutation",
    "feature_importance",
    data_source="test",
    method="permutation",
),
self.pd(
    "feature_importance.mean_decrease_impurity",
    "Feature importance - Mean Decrease Impurity (MDI)",
    "feature_importance",
    method="mean_decrease_impurity",
),
self.pd(
    "feature_importance.coefficients",
    "Feature importance - Coefficients",
    "feature_importance",
    method="coefficients",
),
