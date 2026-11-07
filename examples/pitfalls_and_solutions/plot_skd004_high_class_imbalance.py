"""
.. _example_skd004_high_class_imbalance:

SKD004 - High class imbalance
=============================

:ref:`SKD004 <skd004-high-class-imbalance>` flags a binary classification task
when the majority class exceeds 80 % of rows. Accuracy can look high while
the minority class is ignored as a default. This notebook is
mostly about how to work with that imbalance once the check fires: we do not
try to make SKD004 disappear, because natural prevalence is often the right
thing to keep.

What we do instead (see also :ref:`automated_checks`):

- report absolute counts as well as percentages,
- evaluate ranking and calibration (ROC AUC, log-loss) before trusting
  thresholded precision / recall,
- tune the decision threshold under an explicit precision / recall or cost
  constraint (for example with
  :class:`~sklearn.model_selection.TunedThresholdClassifierCV`),
- avoid ``class_weight`` and resampling when calibrated probabilities matter,
- correct for prevalence shift if you collect minority-only data.

See also:
https://probabl-ai.github.io/calibration-cost-sensitive-learning/content/notebooks/imbalanced_classification.html

We use Covertype forest types 2 (majority) vs 5 (minority) on an 8,000-row
stratified subsample. The goal is to keep natural prevalence, judge probability
quality first, then choose a cut-off that matches the precision / recall
trade-off you care about.
"""

# %%
# Load Covertype (types 2 vs 5)
# =============================
#
# Types 2 vs 5 give a natural imbalance (type 2 is the majority class). We keep
# minority type 5 as the positive class and draw an 8,000-row stratified
# subsample so the gallery stays fast while absolute minority counts remain
# large enough to learn from.

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

df = fetch_covtype(as_frame=True).frame
pair = df.query("Cover_Type.isin([2, 5])")
y_full = (pair["Cover_Type"] == 5).astype(int).rename("is_type_5")
X_full = pair.drop(columns=["Cover_Type"])

X, _, y, _ = train_test_split(
    X_full,
    y_full,
    train_size=8_000,
    stratify=y_full,
    random_state=42,
)

# %%
y.value_counts(normalize=True).round(4)

# %%
y.value_counts()

# %%
# Inspect the feature matrix with :class:`~skrub.TableReport`.

from skrub import TableReport

TableReport(X)

# %%
# The binary target marks type-5 stands; the majority class exceeds 80 % of
# rows, so SKD004 will fire.

TableReport(y)

# %%
# Trigger SKD004 - default classifier on imbalanced labels
# ========================================================
#
# A default gradient boosting classifier does not change label counts. The
# check cares about the class mix in the data, not about whether we reweighted
# the loss.

from sklearn.ensemble import HistGradientBoostingClassifier
from skore import TrainTestSplit, evaluate

splitter = TrainTestSplit(test_size=0.2, random_state=42, stratify=y)
classifier = HistGradientBoostingClassifier(random_state=42)

report = evaluate(
    classifier,
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)
report

# %%
# SKD004 should fire: the majority class exceeds 80 % of rows. Ignore SKD008 to
# avoid correlated-feature warnings from Covertype's constant soil one-hots.

report.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
# Accuracy and F1 are poor defaults under imbalance
# =================================================
#
# Accuracy can be inflated by nearly always predicting the majority class. F1
# averages precision and recall into one number and hides which side of the
# trade-off you care about, so we do not use it here. At the default
# probability cut-off of 0.5, minority recall is often weak because rare
# events receive small predicted probabilities.

report.metrics.summarize(
    metric=["accuracy", "precision", "recall", "roc_auc", "log_loss"],
    data_source="both",
).frame()

# %%
# The stratified hold-out still has only a few dozen type-5 rows against about
# 1,500 majority rows. Most of those scarce positives are predicted as
# majority, so minority recall is low while accuracy stays high.

report.metrics.confusion_matrix().plot()

# %%
# Check ranking and calibration first
# ===================================
#
# Before touching thresholds, ask whether probabilities are any good:
#
# - ROC AUC asks whether positives tend to get higher scores than negatives
#   (threshold-free ranking),
# - log-loss penalizes confident wrong probabilities,
# - a calibration curve asks whether predicted probabilities match observed
#   frequencies.
#
# On the calibration plot, bins of predicted probability are compared to the
# fraction of true positives in each bin. A useful curve hugs the diagonal: when
# the model says "20 %", about 20 % of those rows really are positive. Points
# above the diagonal mean under-confidence (events happen more often than
# predicted); points below mean over-confidence (the model is too sure). With a
# rare class, almost all mass sits at low probabilities, so the curve often
# only appears on the left of the plot; that is expected, not a plotting bug.
#
# If ranking and calibration look reasonable, the model may already be useful;
# the default 0.5 cut-off is simply the wrong operating point for a rare class.
# The next section shows how ``class_weight="balanced"`` can push the curve
# below the diagonal by inflating minority probabilities.

report.metrics.summarize(
    metric=["roc_auc", "log_loss"],
    data_source="test",
).frame()

# %%
report.inspection.calibration_curve(data_source="test", n_bins=10).plot()

# %%
# Class weights as a cautionary comparison
# ========================================
#
# A common reflex is ``class_weight="balanced"``. Rebalancing with weights is
# equivalent in spirit to resampling methods such as SMOTE or random
# oversampling / undersampling: they change the effective class mix and will
# suffer from the same issues. That often improves precision / recall at 0.5
# because it inflates minority probabilities, but it typically breaks
# calibration: predicted probabilities run ahead of observed rates, so the
# curve drifts below the diagonal (over-confidence on the originally rare
# class). If you later recalibrate, the thresholded gains often disappear. We
# show the comparison, then leave weights behind when calibrated probabilities
# matter.

from skore import compare

report_weighted = evaluate(
    HistGradientBoostingClassifier(class_weight="balanced", random_state=42),
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

comparison_weights = compare(
    {"default": report, "class_weight_balanced": report_weighted}
)
comparison_weights.metrics.summarize(
    metric=["precision", "recall", "roc_auc", "log_loss"],
    data_source="test",
).frame()

# %%
# After reweighting, compare this curve to the default one: points tend to sit
# further below the diagonal (over-confident on the rare class).

report_weighted.inspection.calibration_curve(data_source="test", n_bins=10).plot()

# %%
# Tune the decision threshold
# ===========================
#
# Keep the default, prevalence-correct model and change only the decision rule.
# For this demo we require at least 30 % precision on type 5, then maximize
# recall. That floor is an explicit product choice: high enough to limit false
# alarms, low enough that a rare-event model can still catch a useful share of
# true type-5 stands. Replace 0.3 with a cost or capacity constraint in real
# work.
#
# :class:`~sklearn.model_selection.TunedThresholdClassifierCV` searches the
# cut-off by cross-validation and does not change ``predict_proba``, so
# calibration stays intact.

from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import TunedThresholdClassifierCV


def recall_with_min_precision(y_true, y_pred, precision_level=0.3):
    """Maximize recall only among thresholds that keep precision high enough."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    if precision < precision_level:
        return -np.inf
    return recall


threshold_scoring = make_scorer(recall_with_min_precision, precision_level=0.3)

tuned = TunedThresholdClassifierCV(
    estimator=HistGradientBoostingClassifier(random_state=42),
    scoring=threshold_scoring,
    cv=3,
    n_jobs=4,
)

report_tuned = evaluate(
    tuned,
    X=X,
    y=y,
    pos_label=1,
    splitter=splitter,
)

# %%
# SKD004 still fires: label counts did not change. That is expected. We
# improved how we decide, not the histogram SKD004 reads. Ignore SKD008 to
# avoid correlated-feature warnings from Covertype's constant soil one-hots.

report_tuned.checks.summarize(fast_mode=True, ignore=["SKD008"])

# %%
print("Chosen decision threshold:", float(report_tuned.estimator_.best_threshold_))

report_tuned.metrics.summarize(
    metric=["accuracy", "precision", "recall", "roc_auc", "log_loss"],
    data_source="both",
).frame()

# %%
report_tuned.metrics.confusion_matrix().plot()

# %%
# Compare the default 0.5 cut-off to the tuned threshold. Same underlying
# probabilities; only the hard predictions change. Precision / recall move;
# ROC AUC and log-loss stay essentially the same.
#
# The scorer asks for precision of at least 0.3, then maximizes recall: catch
# more type-5 stands without too many false alarms (fraud review, maintenance
# tickets, medical triage). Preferring high precision instead fits cases where
# a false alarm is costly: auto-blocking users, expensive tests, or limited
# outreach budgets.

comparison_thresholds = compare(
    {
        "default_threshold_0.5": report,
        "tuned_threshold": report_tuned,
    }
)
comparison_thresholds.metrics.summarize(
    metric=["precision", "recall", "roc_auc", "log_loss"],
    data_source="test",
).frame()

# %%
# Inspect the precision-recall curve
# ==================================
#
# The dashed line is our precision floor (0.3). The tuned threshold should land
# near the highest-recall point that still respects that floor.

threshold = float(report_tuned.estimator_.best_threshold_)

display = report.metrics.precision_recall()
fig = display.plot()
ax = fig.axes[0]
ax.axhline(0.3, linestyle="--", color="gray", label="precision floor 0.3")
ax.axvline(
    report_tuned.metrics.recall(),
    linestyle=":",
    color="C1",
    label=f"recall at threshold={threshold:.3f}",
)
ax.legend(loc="best")
ax.set_title("Precision-recall curve (test fold)")
fig

# %%
# Collecting more minority data
# =============================
#
# Gathering more type-5 plots can help the model see the rare class. If
# acquisition preferentially samples minority rows, train prevalence no longer
# matches production. Probabilities and thresholds fitted on that mix will be
# biased unless you correct for the shift. Clearing SKD004 by stuffing
# minority rows into the table is therefore not automatically a success.

# %%
# Conclusion
# ==========
#
# SKD004 warns that one class dominates the table. Keep natural prevalence when
# you need honest probabilities; move the threshold when you need a different
# precision / recall trade-off. Class weights and resampling are risky
# shortcuts if calibration matters for the decisions you deploy.
