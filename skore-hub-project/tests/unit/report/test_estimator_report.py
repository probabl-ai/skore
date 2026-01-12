from io import BytesIO

from joblib import dump, hash
from pydantic import ValidationError
from pytest import fixture, mark, raises
from skore import CrossValidationReport, EstimatorReport

from skore_hub_project import Project
from skore_hub_project.artifact.media import (
    EstimatorHtmlRepr,
    MeanDecreaseImpurity,
    PermutationTest,
    PermutationTrain,
    PrecisionRecallTest,
    PrecisionRecallTrain,
    RocTest,
    RocTrain,
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


def serialize(object: EstimatorReport | CrossValidationReport) -> tuple[bytes, str]:
    reports = [object] + getattr(object, "estimator_reports_", [])
    caches = [report_to_clear._cache for report_to_clear in reports]

    object.clear_cache()

    try:
        with BytesIO() as stream:
            dump(object, stream)
            pickle_bytes = stream.getvalue()
    finally:
        for report, cache in zip(reports, caches, strict=True):
            report._cache = cache

    return pickle_bytes, f"skore-{object.__class__.__name__}-{object._hash}"


@fixture
def project():
    return Project("<tenant>", "<name>")


@fixture
def payload(project, binary_classification):
    # Force the compute of the permutations
    binary_classification.feature_importance.permutation(data_source="train", seed=42)
    binary_classification.feature_importance.permutation(data_source="test", seed=42)

    return EstimatorReportPayload(
        project=project,
        report=binary_classification,
        key="<key>",
    )


class TestEstimatorReportPayload:
    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    @mark.usefixtures("monkeypatch_upload_with_mock")
    def test_pickle(
        self, binary_classification, project, payload, upload_mock, respx_mock
    ):
        pickle, checksum = serialize(binary_classification)

        # Ensure checksum is well constructed
        assert payload.pickle.checksum == checksum

        # ensure `upload` is well called
        assert upload_mock.called
        assert not upload_mock.call_args.args
        assert upload_mock.call_args.kwargs.pop("pool")
        assert upload_mock.call_args.kwargs == {
            "project": project,
            "filepath": payload.pickle.filepath,
            "checksum": checksum,
            "content_type": "application/octet-stream",
        }

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

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    def test_medias(self, payload):
        assert list(map(type, payload.medias)) == [
            EstimatorHtmlRepr,
            MeanDecreaseImpurity,
            PermutationTest,
            PermutationTrain,
            PrecisionRecallTest,
            PrecisionRecallTrain,
            RocTest,
            RocTrain,
            TableReportTest,
            TableReportTrain,
        ]

    @mark.usefixtures("monkeypatch_artifact_hub_client")
    @mark.usefixtures("monkeypatch_upload_routes")
    def test_model_dump(self, binary_classification, payload):
        _, checksum = serialize(binary_classification)

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

    def test_exception(self):
        with raises(ValidationError):
            EstimatorReportPayload(
                project=Project("<tenant>", "<name>"), report=None, key="<key>"
            )
