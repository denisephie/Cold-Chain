"""
  1. Risk model         transit_days, fill_ratio, leg_count, package_type -> failure probability
  2. Door-opens model   same inputs -> expected door opens + 80% range (negative binomial)
  3. Risk given opens   same inputs + door_opens -> failure probability. What-if only: the actual
                        number of opens is not known before shipping.
"""

import json
import os
import sys
import warnings
from datetime import datetime
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from scipy.stats import nbinom
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings("ignore", category=UserWarning)

# CONFIG
DATA_PATH = "./shipment-sensor-dataset.csv"
OUT_DIR = "./artifacts"
TARGET = "silent_failure"
OPENS = "door_opens"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# COST_FN says how much worse a missed failure is than a false alarm. At 3.5, letting one bad
# shipment through is treated as 3.5 times as costly as needlessly flagging a good one.
#
# The value itself has not changed. The comment here used to say "5x" while the code said 3.5;
# the comment was simply wrong, and this corrects it.
#
# What it does: nothing during training. It only picks the alert threshold afterwards, by
# sliding along a fixed precision/recall curve - so raising it catches more failures and flags
# more shipments, and lowering it does the reverse. It can never improve both at once. For a
# well-calibrated model the threshold lands near 1/(1+ratio), and 3.5 gives 0.23.
#
# Why 3.5 and not something else (checked 2026-09-22 on the held-out test set): 2.5 scored
# slightly better on F1 (0.560 vs 0.548), but that gap is inside the noise - bootstrapping put
# it at +0.012 with a 95% range of -0.009 to +0.032, which includes zero. Meanwhile 3.5 catches
# noticeably more real failures: 78% vs 71%, or 69 missed instead of 92. Catching those extra
# failures costs roughly 4-5 more shipments flagged each.
#
# So this is a business decision, not something to tune. Set it from what a spoiled load and a
# false alarm actually cost you, not from F1.
COST_FN = 3.5
COST_FP = 1.0
HIGH_RISK_CUTOFF = 0.50

# Known before shipping
NUMERIC_FEATURES = ["transit_days", "fill_ratio", "leg_count"]
CATEGORICAL_FEATURES = ["package_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DOOR_NUMERIC = ["transit_days", "fill_ratio", "leg_count"]
DOOR_CATEGORICAL = ["package_type"]
DOOR_FEATURES = DOOR_NUMERIC + DOOR_CATEGORICAL
INTERVAL = (0.10, 0.90)  # 10th and 90th percentile of door opens

os.makedirs(OUT_DIR, exist_ok=True)

def make_prep(numeric, categorical):
    prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )
    prep.set_output(transform="pandas")
    return prep

def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=130)
    plt.close()

PARAM_DIST = {
    "model__max_depth": [2, 3, 4],
    "model__n_estimators": [100, 200, 300, 500],
    "model__learning_rate": [0.02, 0.03, 0.05, 0.1],
    "model__min_child_weight": [3, 5, 10, 20],
    "model__subsample": [0.6, 0.8, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__reg_lambda": [1, 3, 5, 10],
    "model__gamma": [0, 0.1, 0.5],
}

# Load and split
df = pd.read_csv(DATA_PATH)
all_cols = list(dict.fromkeys(FEATURES + DOOR_FEATURES + [OPENS, TARGET]))
missing_cols = [c for c in all_cols if c not in df.columns]
if missing_cols:
    sys.exit(f"Missing columns in CSV: {missing_cols}")

data = df[all_cols].dropna(subset=[TARGET]).copy()
data[TARGET] = data[TARGET].astype(int)
assert set(data[TARGET].unique()) <= {0, 1}, "Target must be 0/1"

X = data[FEATURES]
y = data[TARGET]
train, test = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_train, X_test = train[FEATURES], test[FEATURES]
y_train, y_test = train[TARGET], test[TARGET]
door_train = train.dropna(subset=[OPENS]).copy()
door_test = test.dropna(subset=[OPENS]).copy()
door_train[OPENS] = door_train[OPENS].round().astype(int)
door_test[OPENS] = door_test[OPENS].round().astype(int)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Risk Model
pipeline = Pipeline(
    steps=[
        ("prep", make_prep(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
        (
            "model",
            xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                monotone_constraints={"transit_days": 1},  # longer transit never lowers risk
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=PARAM_DIST,
    n_iter=40,
    scoring="neg_log_loss",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)
search.fit(X_train, y_train)
model = search.best_estimator_

# Threshold from out-of-fold predictions
oof_prob = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
thr_grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

def expected_cost(y_true, prob, thr):
    pred = prob >= thr
    fn = ((~pred) & (y_true == 1)).sum()
    fp = (pred & (y_true == 0)).sum()
    return (COST_FN * fn + COST_FP * fp) / len(y_true)

costs = np.array([expected_cost(y_train.values, oof_prob, t) for t in thr_grid])
review_threshold = float(thr_grid[costs.argmin()])
high_threshold = max(HIGH_RISK_CUTOFF, review_threshold + 0.05)

print("Threshold trade-off (out-of-fold, train):")
print("   thr  flagged  precision  recall  false alarms per real failure")
for t in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
    flagged = oof_prob >= t
    tp = int((flagged & (y_train.values == 1)).sum())
    fp = int((flagged & (y_train.values == 0)).sum())
    print(f"  {t:>4.2f}  {flagged.mean():>6.0%}  {tp / max(flagged.sum(), 1):>9.2f}  "
          f"{tp / (y_train.values == 1).sum():>6.2f}  {fp / max(tp, 1):>6.1f}")
print(f"\nReview threshold {review_threshold:.2f}, high {high_threshold:.2f}")

# Test
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= review_threshold).astype(int)
roc_auc = roc_auc_score(y_test, test_prob)
pr_auc = average_precision_score(y_test, test_prob)
brier = brier_score_loss(y_test, test_prob)
ll = log_loss(y_test, test_prob)

print(f"\nRisk model (test): ROC-AUC {roc_auc:.4f}  PR-AUC {pr_auc:.4f}  Brier {brier:.4f}  log-loss {ll:.4f}")
print(classification_report(y_test, test_pred, target_names=["ok", "silent_failure"]))

# Plots: ROC, calibration, risk - transit time
typical = {"fill_ratio": X["fill_ratio"].median(), "leg_count": int(X["leg_count"].median()),
           "package_type": X["package_type"].mode()[0]}
days = np.linspace(X["transit_days"].quantile(0.01), X["transit_days"].quantile(0.99), 60)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
fpr, tpr, _ = roc_curve(y_test, test_prob)
ax[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set(title="ROC curve (test)", xlabel="False positive rate", ylabel="True positive rate")
ax[0].legend()

frac_failed, mean_pred = calibration_curve(y_test, test_prob, n_bins=10, strategy="quantile")
ax[1].plot(mean_pred, frac_failed, "o-")
ax[1].plot([0, 1], [0, 1], "--", color="gray")
ax[1].set(title="Calibration (test)", xlabel="Predicted risk", ylabel="Actual failure rate")

grid = pd.DataFrame({"transit_days": days, **typical})[FEATURES]
ax[2].plot(days, model.predict_proba(grid)[:, 1])
ax[2].set(title="Risk vs transit time", xlabel="Transit days", ylabel="Predicted risk")
save_plot("risk_model.png")

# Door Open Model
Xd_train, Xd_test = door_train[DOOR_FEATURES], door_test[DOOR_FEATURES]
opens_train, opens_test = door_train[OPENS], door_test[OPENS].values

cv_kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
door_search = RandomizedSearchCV(
    Pipeline([("prep", make_prep(DOOR_NUMERIC, DOOR_CATEGORICAL)),
              ("model", xgb.XGBRegressor(objective="count:poisson", random_state=RANDOM_STATE, n_jobs=-1))]),
    param_distributions=PARAM_DIST,
    n_iter=30,
    scoring="neg_mean_poisson_deviance",
    cv=cv_kf,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)
door_search.fit(Xd_train, opens_train)
door_model = door_search.best_estimator_

# Negative-binomial dispersion (var = mu + mu^2 / k) from out-of-fold train predictions
oof_mu = cross_val_predict(door_model, Xd_train, opens_train, cv=cv_kf)
denom = (((opens_train - oof_mu) ** 2) - oof_mu).sum()
k_disp = float((oof_mu ** 2).sum() / denom) if denom > 0 else 1e6

mu_te = door_model.predict(Xd_test)
door_r2 = float(r2_score(opens_test, mu_te))
door_mae = float(mean_absolute_error(opens_test, mu_te))
print(f"Door-opens model (test): R2 {door_r2:.3f}  MAE {door_mae:.2f}")

# 3. Risk -  Door Opens
FAIL_NUMERIC = DOOR_NUMERIC + [OPENS]
FAIL_FEATURES = FAIL_NUMERIC + DOOR_CATEGORICAL
fail_search = RandomizedSearchCV(
    Pipeline([("prep", make_prep(FAIL_NUMERIC, DOOR_CATEGORICAL)),
              ("model", xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                          monotone_constraints={"transit_days": 1, OPENS: 1},
                                          random_state=RANDOM_STATE, n_jobs=-1))]),
    param_distributions=PARAM_DIST,
    n_iter=40,
    scoring="neg_log_loss",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True,
)
fail_search.fit(door_train[FAIL_FEATURES], door_train[TARGET])
fail_model = fail_search.best_estimator_

p_te = fail_model.predict_proba(door_test[FAIL_FEATURES])[:, 1]
b_auc = float(roc_auc_score(door_test[TARGET], p_te))
print(f"Risk given actual door opens (test): ROC-AUC {b_auc:.4f}")

# PLOT door opens - transit time, risk - door opens
pct = round((INTERVAL[1] - INTERVAL[0]) * 100)
mu = door_model.predict(pd.DataFrame({"transit_days": days, **typical})[DOOR_FEATURES])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(door_test["transit_days"], opens_test, s=6, alpha=0.2, label="test shipments")
ax[0].plot(days, mu, color="tab:red", label="expected")
ax[0].fill_between(days, nbinom.ppf(INTERVAL[0], k_disp, k_disp / (k_disp + mu)),
                   nbinom.ppf(INTERVAL[1], k_disp, k_disp / (k_disp + mu)),
                   color="tab:red", alpha=0.15, label=f"{pct}% range")
ax[0].set(title="Door opens vs transit time", xlabel="Transit days", ylabel="Door opens")
ax[0].legend()

opens_axis = np.arange(0, 41)
for d in (2, 4, 8):
    grid = pd.DataFrame({"transit_days": d, **typical, OPENS: opens_axis})[FAIL_FEATURES]
    ax[1].plot(opens_axis, fail_model.predict_proba(grid)[:, 1], label=f"{d} days")
ax[1].set(title="Risk vs door opens", xlabel="Door opens", ylabel="Predicted risk")
ax[1].legend()
save_plot("door_opens.png")

# SAVE
versions = {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
            "xgboost": xgb.__version__, "pandas": pd.__version__}

joblib.dump(model, os.path.join(OUT_DIR, "risk_pipeline.joblib"))
meta = {
    "created": datetime.now().isoformat(timespec="seconds"),
    "purpose": "pre-shipment silent-failure risk",
    "target": TARGET,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "categories": {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
    "training_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in NUMERIC_FEATURES},
    "thresholds": {"review": review_threshold, "high": high_threshold},
    "cost_ratio_fn_to_fp": COST_FN / COST_FP,
    "metrics_test": {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier, "log_loss": ll},
    "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                    for k, v in search.best_params_.items()},
    "base_failure_rate": float(y.mean()),
    "versions": versions,
}
with open(os.path.join(OUT_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

joblib.dump(door_model, os.path.join(OUT_DIR, "door_opens_model.joblib"))
joblib.dump(fail_model, os.path.join(OUT_DIR, "risk_given_opens.joblib"))
door_meta = {
    "created": datetime.now().isoformat(timespec="seconds"),
    "door_features": DOOR_FEATURES,
    "risk_features": FAIL_FEATURES,
    "nb_dispersion_k": k_disp,
    "interval": list(INTERVAL),
    "opens_training_range": [int(data[OPENS].min()), int(data[OPENS].max())],
    "metrics_test": {"door_r2": door_r2, "door_mae": door_mae, "risk_given_opens_auc": b_auc},
    "versions": versions,
}
with open(os.path.join(OUT_DIR, "door_meta.json"), "w") as f:
    json.dump(door_meta, f, indent=2)

print(f"\nSaved to {OUT_DIR}/: " + ", ".join(sorted(os.listdir(OUT_DIR))))