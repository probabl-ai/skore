from json import dumps

import sklearn.svm
import skops.io
from pytest import raises, skip
from skore_remote_project.item import SklearnBaseEstimatorItem
from skore_remote_project.item.item import ItemTypeError, bytes_to_b64_str


class Estimator(sklearn.svm.SVC):
    pass


class TestSklearnBaseEstimatorItem:
    def test_factory(self, monkeypatch):
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

    def test_factory_exception(self):
        with raises(ItemTypeError):
            SklearnBaseEstimatorItem.factory(None)

    def test_parameters(self, monkeypatch):
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
        item_parameters = item.__parameters__

        assert item_parameters == {
            "parameters": {
                "class": "SklearnBaseEstimatorItem",
                "parameters": {
                    "estimator_html_repr": estimator_html_repr,
                    "estimator_skops_b64_str": estimator_skops_b64_str,
                    "estimator_skops_untrusted_types": estimator_skops_untrusted_types,
                },
            }
        }

        # Ensure parameters are JSONable
        dumps(item_parameters)

    def test_raw(self):
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
        )

        assert isinstance(item1.__raw__, sklearn.svm.SVC)
        assert isinstance(item2.__raw__, sklearn.svm.SVC)

    def test_raw_untrusted(self):
        estimator = Estimator()
        estimator_skops_bytes = skops.io.dumps(estimator)
        estimator_skops_b64_str = bytes_to_b64_str(estimator_skops_bytes)
        estimator_skops_untrusted_types = skops.io.get_untrusted_types(
            data=estimator_skops_bytes
        )

        if not estimator_skops_untrusted_types:
            skip(
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
        )

        assert isinstance(item1.__raw__, Estimator)
        assert isinstance(item2.__raw__, Estimator)
