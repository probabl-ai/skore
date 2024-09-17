import pytest
import sklearn.svm
import skops.io
from skore.item import SklearnBaseEstimatorItem


class TestSklearnBaseEstimatorItem:
    @pytest.fixture(autouse=True)
    def monkeypatch_datetime(self, monkeypatch, MockDatetime):
        monkeypatch.setattr("skore.item.item.datetime", MockDatetime)

    @pytest.mark.order(0)
    def test_factory(self, monkeypatch, mock_nowstr):
        monkeypatch.setattr("skops.io.dumps", lambda _: "<estimator_skops>")
        monkeypatch.setattr(
            "sklearn.utils.estimator_html_repr", lambda _: "<estimator_html_repr>"
        )

        estimator = sklearn.svm.SVC()
        estimator_skops = "<estimator_skops>"
        estimator_html_repr = "<estimator_html_repr>"

        item = SklearnBaseEstimatorItem.factory(estimator)

        assert item.estimator_skops == estimator_skops
        assert item.estimator_html_repr == estimator_html_repr
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_estimator(self, mock_nowstr):
        estimator = sklearn.svm.SVC()
        estimator_skops = skops.io.dumps(estimator)
        estimator_html_repr = "<estimator_html_repr>"

        item1 = SklearnBaseEstimatorItem.factory(estimator)
        item2 = SklearnBaseEstimatorItem(
            estimator_skops=estimator_skops,
            estimator_html_repr=estimator_html_repr,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert isinstance(item1.estimator, sklearn.svm.SVC)
        assert isinstance(item2.estimator, sklearn.svm.SVC)
