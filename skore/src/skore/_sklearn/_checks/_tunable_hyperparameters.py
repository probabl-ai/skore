from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skore._sklearn._checks._utils import ClassName, ParameterName

# Parameters that serve the same purpose (e.g. tree complexity regularization).
# If the user tunes any member of a group, the others are considered covered.
# When several members of a group are missing, they are collapsed to a single
# suggestion: the first member that is actually missing for the estimator.
EQUIVALENT_PARAM_GROUPS: list[tuple[ParameterName, ...]] = [
    ("min_samples_leaf", "max_depth", "min_samples_split", "max_leaf_nodes"),
    ("l1_ratio", "penalty"),
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
# Primary sources:
#  - Probst, Boulesteix & Bischl (2019), "Tunability: Importance of Hyperparameters
#    of Machine Learning Algorithms", JMLR 20:1-32. (Table 1, Table 3)
#  - van Rijn & Hutter (2018), "Hyperparameter Importance Across Datasets", KDD '18.
#    (Tables 1-3, fANOVA results)
# Secondary: auto-sklearn search spaces, scikit-learn docs, Bergstra & Bengio (2012).
#
# Excluded by design:
#  - Budget / cost-performance params (n_estimators, max_iter, n_iter, n_init,
#    n_restarts_optimizer, max_iter_predict): set as high as affordable.
#  - Low-impact params (bootstrap, criterion, fit_prior, norm,
#    early_exaggeration, init, rotation): rarely worth tuning per literature.
#  - Mode switches / problem specs (novelty, cv, target_type, output_distribution,
#    direction, scoring, score_func, encode, include_bias, interaction_only,
#    initial_strategy): design choices, not tuning axes.
#  - Feature selection estimators: their params (k, percentile, n_features_to_select,
#    step, threshold) are design choices about the procedure, not model tuning.
#  - Clustering estimators: not supported by skore reports.
TUNABLE_HYPERPARAMETERS: dict[ClassName, set[ParameterName]] = {
    # ===== LINEAR MODELS =====
    "Ridge": {"alpha"},
    "RidgeClassifier": {"alpha"},
    "Lasso": {"alpha"},
    "ElasticNet": {"alpha", "l1_ratio"},
    "LogisticRegression": {"C", "penalty", "l1_ratio"},
    "SGDClassifier": {
        "alpha",
        "loss",
        "penalty",
        "l1_ratio",
        "learning_rate",
        "eta0",
    },
    "SGDRegressor": {
        "alpha",
        "loss",
        "penalty",
        "l1_ratio",
        "learning_rate",
        "eta0",
    },
    "HuberRegressor": {"alpha", "epsilon"},
    "QuantileRegressor": {"alpha"},
    "PoissonRegressor": {"alpha"},
    "GammaRegressor": {"alpha"},
    "TweedieRegressor": {"alpha", "power"},
    # ===== TREES =====
    "DecisionTreeClassifier": {
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "ccp_alpha",
    },
    "DecisionTreeRegressor": {
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "ccp_alpha",
    },
    # ===== RANDOM FORESTS / EXTRA TREES =====
    "RandomForestClassifier": {
        "max_features",
        "min_samples_leaf",
        "max_depth",
        "max_samples",
    },
    "RandomForestRegressor": {
        "max_features",
        "min_samples_leaf",
        "max_depth",
        "max_samples",
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
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "max_features",
    },
    "GradientBoostingRegressor": {
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "max_features",
        "loss",
    },
    "HistGradientBoostingClassifier": {
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "max_depth",
    },
    "HistGradientBoostingRegressor": {
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "max_depth",
        "loss",
    },
    "AdaBoostClassifier": {"learning_rate", "estimator"},
    "AdaBoostRegressor": {"learning_rate", "estimator", "loss"},
    "BaggingClassifier": {"max_samples", "max_features"},
    "BaggingRegressor": {"max_samples", "max_features"},
    "IsolationForest": {"max_samples", "contamination", "max_features"},
    # ===== SVMs =====
    "SVC": {"C", "kernel", "gamma", "degree", "coef0"},
    "SVR": {"C", "kernel", "gamma", "degree", "coef0", "epsilon"},
    "LinearSVC": {"C", "penalty", "loss"},
    "LinearSVR": {"C", "loss", "epsilon"},
    "NuSVC": {"nu", "kernel", "gamma", "degree", "coef0"},
    "NuSVR": {"nu", "C", "kernel", "gamma", "degree"},
    "OneClassSVM": {"nu", "kernel", "gamma"},
    # ===== NEIGHBORS =====
    "KNeighborsClassifier": {"n_neighbors", "weights", "p", "metric"},
    "KNeighborsRegressor": {"n_neighbors", "weights", "p", "metric"},
    "RadiusNeighborsClassifier": {"radius", "weights", "p"},
    "RadiusNeighborsRegressor": {"radius", "weights", "p"},
    "LocalOutlierFactor": {"n_neighbors", "contamination"},
    # ===== NAIVE BAYES =====
    "MultinomialNB": {"alpha"},
    "BernoulliNB": {"alpha", "binarize"},
    "ComplementNB": {"alpha"},
    # ===== DISCRIMINANT ANALYSIS =====
    "LinearDiscriminantAnalysis": {"solver", "shrinkage"},
    "QuadraticDiscriminantAnalysis": {"reg_param"},
    # ===== DECOMPOSITION =====
    "PCA": {"n_components", "whiten"},
    "KernelPCA": {"n_components", "kernel", "gamma", "degree", "coef0"},
    "SparsePCA": {"n_components", "alpha", "ridge_alpha"},
    "TruncatedSVD": {"n_components"},
    "NMF": {"n_components", "alpha_W", "alpha_H", "l1_ratio"},
    "FactorAnalysis": {"n_components"},
    "LatentDirichletAllocation": {
        "n_components",
        "doc_topic_prior",
        "topic_word_prior",
        "learning_decay",
        "learning_offset",
    },
    # ===== MANIFOLD =====
    "TSNE": {"perplexity", "learning_rate"},
    "Isomap": {"n_neighbors", "n_components"},
    "LocallyLinearEmbedding": {"n_neighbors", "n_components", "reg", "method"},
    # ===== KERNEL APPROXIMATION =====
    "Nystroem": {"n_components", "kernel", "gamma", "degree", "coef0"},
    "RBFSampler": {"n_components", "gamma"},
    "SkewedChi2Sampler": {"n_components", "skewedness"},
    "AdditiveChi2Sampler": {"sample_steps"},
    "PolynomialCountSketch": {"n_components", "degree", "gamma", "coef0"},
    # ===== PREPROCESSING =====
    "QuantileTransformer": {"n_quantiles"},
    "PowerTransformer": {"method"},
    "KBinsDiscretizer": {"n_bins", "strategy"},
    "PolynomialFeatures": {"degree"},
    "SplineTransformer": {"n_knots", "degree", "knots"},
    "TargetEncoder": {"smooth"},
    # ===== IMPUTATION =====
    "KNNImputer": {"n_neighbors", "weights"},
    "IterativeImputer": {"estimator", "n_nearest_features"},
    # ===== GAUSSIAN PROCESSES =====
    "GaussianProcessClassifier": {"kernel"},
    "GaussianProcessRegressor": {"kernel", "alpha"},
    # ===== NEURAL NETWORKS =====
    "MLPClassifier": {
        "hidden_layer_sizes",
        "alpha",
        "learning_rate_init",
        "activation",
        "batch_size",
        "solver",
    },
    "MLPRegressor": {
        "hidden_layer_sizes",
        "alpha",
        "learning_rate_init",
        "activation",
        "batch_size",
        "solver",
    },
    # ===== CALIBRATION =====
    "CalibratedClassifierCV": {"method"},
}
