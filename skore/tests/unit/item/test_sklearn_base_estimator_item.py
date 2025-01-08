import pytest
import sklearn.svm
import skops.io
from skore.persistence.item import ItemTypeError, SklearnBaseEstimatorItem


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
        estimator_skops = "<estimator_skops>"
        estimator_skops_untrusted_types = "<estimator_skops_untrusted_types>"

        monkeypatch.setattr(
            "sklearn.utils.estimator_html_repr",
            lambda *args, **kwargs: estimator_html_repr,
        )
        monkeypatch.setattr("skops.io.dumps", lambda *args, **kwargs: estimator_skops)
        monkeypatch.setattr(
            "skops.io.get_untrusted_types",
            lambda *args, **kwargs: estimator_skops_untrusted_types,
        )

        item = SklearnBaseEstimatorItem.factory(estimator)

        assert item.estimator_html_repr == estimator_html_repr
        assert item.estimator_skops == estimator_skops
        assert item.estimator_skops_untrusted_types == estimator_skops_untrusted_types
        assert item.created_at == mock_nowstr
        assert item.updated_at == mock_nowstr

    @pytest.mark.order(1)
    def test_estimator(self, mock_nowstr):
        estimator = sklearn.svm.SVC()
        estimator_skops = skops.io.dumps(estimator)
        estimator_skops_untrusted_types = skops.io.get_untrusted_types(
            data=estimator_skops
        )

        item1 = SklearnBaseEstimatorItem.factory(estimator)
        item2 = SklearnBaseEstimatorItem(
            estimator_html_repr=None,
            estimator_skops=estimator_skops,
            estimator_skops_untrusted_types=estimator_skops_untrusted_types,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert isinstance(item1.estimator, sklearn.svm.SVC)
        assert isinstance(item2.estimator, sklearn.svm.SVC)

    @pytest.mark.order(1)
    def test_estimator_untrusted(self, mock_nowstr):
        estimator = Estimator()
        estimator_skops = skops.io.dumps(estimator)
        estimator_skops_untrusted_types = skops.io.get_untrusted_types(
            data=estimator_skops
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
            estimator_skops=estimator_skops,
            estimator_skops_untrusted_types=estimator_skops_untrusted_types,
            created_at=mock_nowstr,
            updated_at=mock_nowstr,
        )

        assert isinstance(item1.estimator, Estimator)
        assert isinstance(item2.estimator, Estimator)

    def test_get_serializable_dict(self, mock_nowstr):
        estimator = Estimator()
        item = SklearnBaseEstimatorItem.factory(estimator)
        serializable = item.as_serializable_dict()

        assert serializable == {
            "updated_at": mock_nowstr,
            "created_at": mock_nowstr,
            "media_type": "application/vnd.sklearn.estimator+html",
            "value": item.estimator_html_repr,
        }
