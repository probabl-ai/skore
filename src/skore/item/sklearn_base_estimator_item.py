from __future__ import annotations

from functools import cached_property

import sklearn
import skops.io


class SklearnBaseEstimatorItem:
    def __init__(self, estimator_skops, estimator_html_repr):
        self.estimator_skops = estimator_skops
        self.estimator_html_repr = estimator_html_repr

    @cached_property
    def estimator(self) -> sklearn.base.BaseEstimator:
        return sklearn.io.loads(self.estimator_skops)

    @property
    def __dict__(self):
        return {
            "estimator_skops": self.estimator_skops,
            "estimator_html_repr": self.estimator_html_repr,
        }

    @classmethod
    def factory(cls, estimator: sklearn.base.BaseEstimator) -> SklearnBaseEstimatorItem:
        instance = cls(
            skops.io.dumps(estimator), sklearn.utils.estimator_html_repr(estimator)
        )

        # add estimator as cached property
        instance.estimator = estimator

        return instance
