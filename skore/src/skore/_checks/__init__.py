"""Checks that can be run on `skore` reports."""

from skore._checks.base import Check, ChecksSummaryDisplay
from skore._checks.skd001_overfitting import CheckOverfitting
from skore._checks.skd002_underfitting import CheckUnderfitting
from skore._checks.skd003_metrics_consistency import CheckMetricsConsistencyAcrossSplits
from skore._checks.skd004_high_class_imbalance import CheckHighClassImbalance
from skore._checks.skd005_underrepresented_classes import CheckUnderrepresentedClasses
from skore._checks.skd006_coefficients_interpretation import (
    CheckCoefficientsInterpretation,
)
from skore._checks.skd007_mdi_high_cardinality_bias import CheckMDIHighCardinalityBias
from skore._checks.skd008_correlated_features import CheckCorrelatedFeatures
from skore._checks.skd009_worse_than_baseline import CheckWorseThanBaseline
from skore._checks.skd010_slower_than_baseline import CheckSlowerThanBaseline
from skore._checks.skd011_golden_feature import CheckGoldenFeature
from skore._checks.skd012_useless_features import CheckUselessFeatures
from skore._checks.skd013_train_test_time_overlap import CheckTrainTestTimeOverlap
from skore._checks.skd014_hyperparams_at_search_edge import CheckHyperparamsAtSearchEdge
from skore._checks.skd015_search_params_to_tune import CheckSearchParamsToTune
from skore._checks.skd016_estimator_not_tuned import CheckEstimatorNotTuned
from skore._checks.utils import CheckNotApplicable

_BUILTIN_CHECKS = [
    CheckOverfitting(),
    CheckUnderfitting(),
    CheckMetricsConsistencyAcrossSplits(),
    CheckHighClassImbalance(),
    CheckUnderrepresentedClasses(),
    CheckCoefficientsInterpretation(),
    CheckMDIHighCardinalityBias(),
    CheckCorrelatedFeatures(),
    CheckWorseThanBaseline(),
    CheckSlowerThanBaseline(),
    CheckGoldenFeature(),
    CheckUselessFeatures(),
    CheckTrainTestTimeOverlap(),
    CheckHyperparamsAtSearchEdge(),
    CheckSearchParamsToTune(),
    CheckEstimatorNotTuned(),
]

__all__ = ["Check", "ChecksSummaryDisplay", "CheckNotApplicable"]
