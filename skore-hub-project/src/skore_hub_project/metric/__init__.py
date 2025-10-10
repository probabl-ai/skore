"""Class definitions of the payloads used to send a metric to ``hub``."""

from .accuracy import (
    AccuracyTest,
    AccuracyTestMean,
    AccuracyTestStd,
    AccuracyTrain,
    AccuracyTrainMean,
    AccuracyTrainStd,
)
from .brier_score import (
    BrierScoreTest,
    BrierScoreTestMean,
    BrierScoreTestStd,
    BrierScoreTrain,
    BrierScoreTrainMean,
    BrierScoreTrainStd,
)
from .log_loss import (
    LogLossTest,
    LogLossTestMean,
    LogLossTestStd,
    LogLossTrain,
    LogLossTrainMean,
    LogLossTrainStd,
)
from .precision import (
    PrecisionTest,
    PrecisionTestMean,
    PrecisionTestStd,
    PrecisionTrain,
    PrecisionTrainMean,
    PrecisionTrainStd,
)
from .r2 import (
    R2Test,
    R2TestMean,
    R2TestStd,
    R2Train,
    R2TrainMean,
    R2TrainStd,
)
from .recall import (
    RecallTest,
    RecallTestMean,
    RecallTestStd,
    RecallTrain,
    RecallTrainMean,
    RecallTrainStd,
)
from .rmse import (
    RmseTest,
    RmseTestMean,
    RmseTestStd,
    RmseTrain,
    RmseTrainMean,
    RmseTrainStd,
)
from .roc_auc import (
    RocAucTest,
    RocAucTestMean,
    RocAucTestStd,
    RocAucTrain,
    RocAucTrainMean,
    RocAucTrainStd,
)
from .timing import (
    FitTime,
    FitTimeMean,
    FitTimeStd,
    PredictTimeTest,
    PredictTimeTestMean,
    PredictTimeTestStd,
    PredictTimeTrain,
    PredictTimeTrainMean,
    PredictTimeTrainStd,
)

# __Payload__
# name: str
# value: float
# data_source: str | None = None

__all__ = [
    "AccuracyTest",
    "AccuracyTestMean",
    "AccuracyTestStd",
    "AccuracyTrain",
    "AccuracyTrainMean",
    "AccuracyTrainStd",
    "BrierScoreTest",
    "BrierScoreTestMean",
    "BrierScoreTestStd",
    "BrierScoreTrain",
    "BrierScoreTrainMean",
    "BrierScoreTrainStd",
    "FitTime",
    "FitTimeMean",
    "FitTimeStd",
    "LogLossTest",
    "LogLossTestMean",
    "LogLossTestStd",
    "LogLossTrain",
    "LogLossTrainMean",
    "LogLossTrainStd",
    "PrecisionTest",
    "PrecisionTestMean",
    "PrecisionTestStd",
    "PrecisionTrain",
    "PrecisionTrainMean",
    "PrecisionTrainStd",
    "PredictTimeTest",
    "PredictTimeTestMean",
    "PredictTimeTestStd",
    "PredictTimeTrain",
    "PredictTimeTrainMean",
    "PredictTimeTrainStd",
    "R2Test",
    "R2TestMean",
    "R2TestStd",
    "R2Train",
    "R2TrainMean",
    "R2TrainStd",
    "RecallTest",
    "RecallTestMean",
    "RecallTestStd",
    "RecallTrain",
    "RecallTrainMean",
    "RecallTrainStd",
    "RmseTest",
    "RmseTestMean",
    "RmseTestStd",
    "RmseTrain",
    "RmseTrainMean",
    "RmseTrainStd",
    "RocAucTest",
    "RocAucTestMean",
    "RocAucTestStd",
    "RocAucTrain",
    "RocAucTrainMean",
    "RocAucTrainStd",
]
