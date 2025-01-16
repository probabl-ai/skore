from sklearn.pipeline import Pipeline


def _check_supported_estimator(supported_estimators):
    def check(accessor):
        estimator = accessor._parent.estimator_
        if isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]
        supported_estimator = isinstance(estimator, supported_estimators)

        if not supported_estimator:
            raise AttributeError(
                f"The {estimator.__class__.__name__} estimator is not supported "
                "by the function called."
            )

        return True

    return check
