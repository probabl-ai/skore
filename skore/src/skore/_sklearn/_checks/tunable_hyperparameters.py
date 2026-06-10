from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skore._sklearn._checks._utils import ClassName, ParameterName

# Parameters that serve the same purpose (e.g. tree complexity regularization).
# If the user tunes any member of a group, the others are considered covered.
# When several members of a group are missing, they are collapsed to a single
# suggestion: the first member that is actually missing for the estimator.
EQUIVALENT_PARAM_GROUPS: list[tuple[ParameterName, ...]] = [
    ("max_leaf_nodes", "min_samples_leaf", "max_depth", "min_samples_split"),
]


# Init params that don't change the learned model in a hyperparameter-tuning
# sense: reproducibility, parallelism, logging, memory, class re-weighting.
# Setting any of these should NOT count as "the estimator has been tuned".
INFRASTRUCTURE_PARAMS: set[ParameterName] = {
    "random_state",
    "n_jobs",
    "verbose",
    "warm_start",
    "class_weight",
    "copy",
    "copy_X",
    "cache_size",
}


# Hyperparameters worth tuning for sklearn estimators.
# Includes hyperparameters that have no obvious default value and that are always
# worth tuning.
HYPERPARAMETERS_TO_TUNE: dict[ClassName, set[ParameterName]] = {
    # ===== LINEAR MODELS =====
    "Ridge": {"alpha"},
    "RidgeClassifier": {"alpha"},
    "Lasso": {"alpha"},
    "ElasticNet": {"alpha", "l1_ratio"},
    "LogisticRegression": {"C"},
    # ===== TREES =====
    "DecisionTreeClassifier": {
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "ccp_alpha",
    },
    "DecisionTreeRegressor": {
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "ccp_alpha",
    },
    # ===== RANDOM FORESTS / EXTRA TREES =====
    "RandomForestClassifier": {
        "max_features",
        "min_samples_leaf",
        "max_depth",
    },
    "RandomForestRegressor": {
        "max_features",
        "min_samples_leaf",
        "max_depth",
    },
    "ExtraTreesClassifier": {
        "max_features",
        "min_samples_leaf",
        "max_depth",
    },
    "ExtraTreesRegressor": {
        "max_features",
        "min_samples_leaf",
        "max_depth",
    },
    # ===== GRADIENT BOOSTING =====
    "GradientBoostingClassifier": {
        "learning_rate",
        "max_leaf_nodes",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    },
    "GradientBoostingRegressor": {
        "learning_rate",
        "max_leaf_nodes",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    },
    "HistGradientBoostingClassifier": {
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "max_depth",
    },
    "HistGradientBoostingRegressor": {
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "max_depth",
    },
    "BaggingClassifier": {"max_samples", "max_features"},
    "BaggingRegressor": {"max_samples", "max_features"},
    # ===== SVMs =====
    "SVC": {"C"},
    "SVR": {"C"},
    "LinearSVC": {"C"},
    "LinearSVR": {"C"},
    # ===== NEIGHBORS =====
    "KNeighborsClassifier": {"n_neighbors"},
    "KNeighborsRegressor": {"n_neighbors"},
    "RadiusNeighborsClassifier": {"radius"},
    "RadiusNeighborsRegressor": {"radius"},
    # ===== DECOMPOSITION =====
    "PCA": {"n_components"},
    "KernelPCA": {"n_components"},
    "SparsePCA": {"n_components"},
    # ===== KERNEL APPROXIMATION =====
    "Nystroem": {"n_components"},
    "RBFSampler": {"n_components", "gamma"},
    "SkewedChi2Sampler": {"n_components", "skewedness"},
    "AdditiveChi2Sampler": {"sample_steps"},
    "PolynomialCountSketch": {"n_components", "degree", "gamma"},
    # ===== PREPROCESSING =====
    "KBinsDiscretizer": {"n_bins"},
    "SplineTransformer": {"n_knots"},
    # ===== IMPUTATION =====
    "KNNImputer": {"n_neighbors"},
}
