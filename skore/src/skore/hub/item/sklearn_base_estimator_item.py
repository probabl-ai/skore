from __future__ import annotations

from typing import TYPE_CHECKING

from .item import (
    Item,
    ItemTypeError,
    Representation,
    b64_str_to_bytes,
    bytes_to_b64_str,
)

if TYPE_CHECKING:
    import sklearn.base


class SklearnBaseEstimatorItem(Item):
    def __init__(
        self,
        estimator_html_repr: str,
        estimator_skops_b64_str: str,
        estimator_skops_untrusted_types: list[str],
    ):
        self.estimator_html_repr = estimator_html_repr
        self.estimator_skops_b64_str = estimator_skops_b64_str
        self.estimator_skops_untrusted_types = estimator_skops_untrusted_types

    @property
    def __raw__(self) -> sklearn.base.BaseEstimator:
        import skops.io

        estimator_skops_bytes = b64_str_to_bytes(self.estimator_skops_b64_str)

        return skops.io.loads(
            data=estimator_skops_bytes,
            trusted=self.estimator_skops_untrusted_types,
        )

    @property
    def __representation__(self) -> Representation:
        return Representation(
            media_type="application/vnd.sklearn.estimator+html",
            value=self.estimator_html_repr,
        )

    @classmethod
    def factory(cls, estimator: sklearn.base.BaseEstimator) -> SklearnBaseEstimatorItem:
        import sklearn.base

        if not isinstance(estimator, sklearn.base.BaseEstimator):
            raise ItemTypeError(f"Type '{estimator.__class__}' is not supported.")

        import sklearn.utils
        import skops.io

        estimator_html_repr = sklearn.utils.estimator_html_repr(estimator)
        estimator_skops_bytes = skops.io.dumps(estimator)
        estimator_skops_b64_str = bytes_to_b64_str(estimator_skops_bytes)
        estimator_skops_untrusted_types = skops.io.get_untrusted_types(
            data=estimator_skops_bytes
        )

        return cls(
            estimator_html_repr,
            estimator_skops_b64_str,
            estimator_skops_untrusted_types,
        )
