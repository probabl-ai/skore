import json

import pytest
import sklearn.svm
import skops.io
from skore.persistence.item import ItemTypeError, SklearnBaseEstimatorItem
from skore.utils import bytes_to_b64_str


class Estimator(sklearn.svm.SVC):
    pass


class TestSklearnBaseEstimatorItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.persistence.item.item.datetime", MockDatetime)

    def test_factory_exception(self):
        with pytest.raises(ItemTypeError):
            SklearnBaseEstimatorItem.factory(None)

    @pytest.mark.order(0)
    def test_factory(self, monkeypatch, mock_nowstr):
        estimator = sklearn.svm.SVC()
        estimator_html_repr = "<estimator_html_repr>"
        estimator_skops_bytes = b"<estimator_skops>"
        estimator_skops_b64_str = bytes_to_b64_str(b"<estimator_skops>")
        estimator_skops_untrusted_types = "<estimator_skops_untrusted_types>"

        monkeypatch.setattr(
            "sklearn.utils.estimator_html_repr",
            lambda *args, **kwargs: estimator_html_repr,
        )
        monkeypatch.setattr(
            "skops.io.dumps", lambda *args, **kwargs: estimator_skops_bytes
        )
        monkeypatch.setattr(
            "skops.io.get_untrusted_types",
            lambda *args, **kwargs: estimator_skops_untrusted_types,
        )

        item = SklearnBaseEstimatorItem.factory(estimator)

        assert item.estimator_html_repr == estimator_html_repr
        assert item.estimator_skops_b64_str == estimator_skops_b64_str
        assert item.estimator_skops_untrusted_types == estimator_skops_untrusted_types
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    def test_ensure_jsonable(self):
        estimator = sklearn.svm.SVC()

        item = SklearnBaseEstimatorItem.factory(estimator)
        item_parameters = item.__parameters__

        json.dumps(item_parameters)

    @pytest.mark.order(1)
    def test_estimator(self, mock_nowstr):
        estimator = sklearn.svm.SVC()
        estimator_skops_bytes = skops.io.dumps(estimator)
        estimator_skops_b64_str = bytes_to_b64_str(estimator_skops_bytes)
        estimator_skops_untrusted_types = skops.io.get_untrusted_types(
            data=estimator_skops_bytes
        )

        item1 = SklearnBaseEstimatorItem.factory(estimator)
        item2 = SklearnBaseEstimatorItem(
            estimator_html_repr=None,
            estimator_skops_b64_str=estimator_skops_b64_str,
            estimator_skops_untrusted_types=estimator_skops_untrusted_types,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert isinstance(item1.estimator, sklearn.svm.SVC)
        assert isinstance(item2.estimator, sklearn.svm.SVC)

    @pytest.mark.order(1)
    def test_estimator_untrusted(self, mock_nowstr):
        estimator = Estimator()
        estimator_skops_bytes = skops.io.dumps(estimator)
        estimator_skops_b64_str = bytes_to_b64_str(estimator_skops_bytes)
        estimator_skops_untrusted_types = skops.io.get_untrusted_types(
            data=estimator_skops_bytes
        )

        if not estimator_skops_untrusted_types:
            pytest.skip(
                """
                This test is only intended to exhaustively test an untrusted estimator.
                The untrusted Estimator class seems to be trusted by default.
                Something changed in `skops`.
                """
            )

        item1 = SklearnBaseEstimatorItem.factory(estimator)
        item2 = SklearnBaseEstimatorItem(
            estimator_html_repr=None,
            estimator_skops_b64_str=estimator_skops_b64_str,
            estimator_skops_untrusted_types=estimator_skops_untrusted_types,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert isinstance(item1.estimator, Estimator)
        assert isinstance(item2.estimator, Estimator)
