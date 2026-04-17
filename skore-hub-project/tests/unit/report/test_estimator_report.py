from io import BytesIO
from unittest.mock import Mock

from joblib import hash, dump
from pydantic import ValidationError
from pytest import fixture, mark, raises
from skore import CrossValidationReport, EstimatorReport

from skore_hub_project.artifact.pickle import Pickle
from skore_hub_project.artifact.media import (
    ConfusionMatrixDataFrameTest,
    ConfusionMatrixDataFrameTrain,
    ConfusionMatrixSVGTest,
    ConfusionMatrixSVGTrain,
    EstimatorHtmlRepr,
    ImpurityDecrease,
    PermutationImportanceTest,
    PermutationImportanceTrain,
    PrecisionRecallDataFrameTest,
    PrecisionRecallDataFrameTrain,
    PrecisionRecallSVGTest,
    PrecisionRecallSVGTrain,
    RocDataFrameTest,
    RocDataFrameTrain,
    RocSVGTest,
    RocSVGTrain,
    TableReportTest,
    TableReportTrain,
)
from skore_hub_project.metric import (
    AccuracyTest,
    AccuracyTrain,
    BrierScoreTest,
    BrierScoreTrain,
    FitTime,
    LogLossTest,
    LogLossTrain,
    PrecisionTest,
    PrecisionTrain,
    PredictTimeTest,
    PredictTimeTrain,
    RecallTest,
    RecallTrain,
    RocAucTest,
    RocAucTrain,
)
from skore_hub_project.report import EstimatorReportPayload


@fixture
def payload(project, binary_classification):
    # Force the compute of the permutations
    binary_classification.inspection.permutation_importance(
        data_source="train", seed=42
    )
    binary_classification.inspection.permutation_importance(data_source="test", seed=42)

    return EstimatorReportPayload(
        project=project,
        report=binary_classification,
        key="<key>",
    )


class TestEstimatorReportPayload:
    @mark.respx()
    def test_pickle(self, tmp_path, payload, project, binary_classification):
        pickle = payload.pickle

        assert type(pickle) is Pickle
        assert pickle.project == project
        assert pickle.report == binary_classification
        assert pickle.computed
        assert pickle.uploaded

        # ensure that there is no residual file
        assert not len(list(tmp_path.iterdir()))

    @mark.respx(assert_all_called=False)
    def test_pickle_uploaded(
        self, monkeypatch, payload, project, binary_classification
    ):
        compute, upload = Mock(), Mock()

        monkeypatch.setattr("skore_hub_project.artifact.pickle.Pickle.compute", compute)
        monkeypatch.setattr("skore_hub_project.artifact.pickle.Pickle.upload", upload)
        monkeypatch.setattr("skore_hub_project.artifact.pickle.Pickle.uploaded", True)

        pickle = payload.pickle

        assert type(pickle) is Pickle
        assert pickle.project == project
        assert pickle.report == binary_classification
        assert not compute.called
        assert not upload.called

    @mark.respx(assert_all_called=False)
    def test_metrics(self, payload):
        assert list(map(type, payload.metrics)) == [
            AccuracyTest,
            AccuracyTrain,
            BrierScoreTest,
            BrierScoreTrain,
            LogLossTest,
            LogLossTrain,
            PrecisionTest,
            PrecisionTrain,
            RecallTest,
            RecallTrain,
            RocAucTest,
            RocAucTrain,
            FitTime,
            PredictTimeTest,
            PredictTimeTrain,
        ]

    @mark.respx(assert_all_called=False)
    def test_metrics_raises_exception(self, monkeypatch, payload):
        """
        Since metrics compute is multi-threaded, ensure that any exceptions thrown in a
        sub-thread are also thrown in the main thread.
        """

        def raise_exception(_):
            raise Exception("test_metrics_raises_exception")

        monkeypatch.setattr(
            "skore_hub_project.report.estimator_report.EstimatorReportPayload.METRICS",
            [AccuracyTest],
        )
        monkeypatch.setattr(
            "skore_hub_project.metric.AccuracyTest.compute", raise_exception
        )

        with raises(Exception, match="test_metrics_raises_exception"):
            list(map(type, payload.metrics))

    @mark.respx()
    def test_medias(self, tmp_path, payload, project, binary_classification):
        types = [
            ConfusionMatrixDataFrameTest,
            ConfusionMatrixDataFrameTrain,
            ConfusionMatrixSVGTest,
            ConfusionMatrixSVGTrain,
            EstimatorHtmlRepr,
            ImpurityDecrease,
            PermutationImportanceTest,
            PermutationImportanceTrain,
            PrecisionRecallDataFrameTest,
            PrecisionRecallDataFrameTrain,
            PrecisionRecallSVGTest,
            PrecisionRecallSVGTrain,
            RocDataFrameTest,
            RocDataFrameTrain,
            RocSVGTest,
            RocSVGTrain,
            TableReportTest,
            TableReportTrain,
        ]

        medias = payload.medias

        assert isinstance(medias, list)
        assert len(medias) == len(types)

        for i, media in enumerate(medias):
            assert type(media) is types[i]
            assert media.project == project
            assert media.report == binary_classification
            assert media.computed
            assert media.uploaded

        # ensure that there is no residual file
        assert not len(list(tmp_path.iterdir()))

    @mark.respx(assert_all_called=False)
    def test_medias_none(self, monkeypatch, payload):
        types = [
            ConfusionMatrixDataFrameTest,
            ConfusionMatrixDataFrameTrain,
            ConfusionMatrixSVGTest,
            ConfusionMatrixSVGTrain,
            EstimatorHtmlRepr,
            ImpurityDecrease,
            PermutationImportanceTest,
            PermutationImportanceTrain,
            PrecisionRecallDataFrameTest,
            PrecisionRecallDataFrameTrain,
            PrecisionRecallSVGTest,
            PrecisionRecallSVGTrain,
            RocDataFrameTest,
            RocDataFrameTrain,
            RocSVGTest,
            RocSVGTrain,
            TableReportTest,
            TableReportTrain,
        ]

        compute, upload = Mock(), Mock()

        for cls in types:
            monkeypatch.setattr(f"{cls.__module__}.{cls.__name__}.checksum", None)
            monkeypatch.setattr(f"{cls.__module__}.{cls.__name__}.compute", compute)
            monkeypatch.setattr(f"{cls.__module__}.{cls.__name__}.upload", upload)

        medias = payload.medias

        assert isinstance(medias, list)
        assert not medias
        assert not compute.called
        assert not upload.called

    @mark.respx(assert_all_called=False)
    def test_medias_uploaded(
        self, monkeypatch, payload, project, binary_classification
    ):
        types = [
            ConfusionMatrixDataFrameTest,
            ConfusionMatrixDataFrameTrain,
            ConfusionMatrixSVGTest,
            ConfusionMatrixSVGTrain,
            EstimatorHtmlRepr,
            ImpurityDecrease,
            PermutationImportanceTest,
            PermutationImportanceTrain,
            PrecisionRecallDataFrameTest,
            PrecisionRecallDataFrameTrain,
            PrecisionRecallSVGTest,
            PrecisionRecallSVGTrain,
            RocDataFrameTest,
            RocDataFrameTrain,
            RocSVGTest,
            RocSVGTrain,
            TableReportTest,
            TableReportTrain,
        ]

        compute, upload = Mock(), Mock()

        for cls in types:
            monkeypatch.setattr(f"{cls.__module__}.{cls.__name__}.checksum", "checksum")
            monkeypatch.setattr(f"{cls.__module__}.{cls.__name__}.compute", compute)
            monkeypatch.setattr(f"{cls.__module__}.{cls.__name__}.upload", upload)
            monkeypatch.setattr(f"{cls.__module__}.{cls.__name__}.uploaded", True)

        medias = payload.medias

        assert isinstance(medias, list)
        assert len(medias) == len(types)
        assert not compute.called
        assert not upload.called

        for i, media in enumerate(medias):
            assert type(media) is types[i]
            assert media.project == project
            assert media.report == binary_classification

    @mark.respx()
    def test_model_dump(self, binary_classification, payload):
        checksum = f"skore-EstimatorReport-{binary_classification.id}"

        payload_dict = payload.model_dump()

        payload_dict.pop("metrics")
        payload_dict.pop("medias")

        assert payload_dict == {
            "key": "<key>",
            "estimator_class_name": "RandomForestClassifier",
            "dataset_fingerprint": hash(binary_classification.y_test),
            "ml_task": "binary-classification",
            "pickle": {
                "checksum": checksum,
                "content_type": "application/octet-stream",
            },
        }

    @mark.respx(assert_all_called=False)
    def test_exception(self, project):
        with raises(ValidationError):
            EstimatorReportPayload(project=project, report=None, key="<key>")
