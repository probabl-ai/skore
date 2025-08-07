from .media import Media, Representation

EstimatorReport = Any
CrossValidationReport = Any
Report = EstimatorReport | CrossValidationReport


class EstimatorHtmlRepr(Media):
    report: Report = Field(repr=False, exclude=True)
    key: Literal["estimator_html_repr"] = "estimator_html_repr"
    category: Literal["model"] = "model"

    @computed_field
    @cached_property
    def representation(self) -> Representation:
        import sklearn.utils

        return Representation(
            media_type="text/html",
            value=sklearn.utils.estimator_html_repr(self.report.estimator_),
        )
