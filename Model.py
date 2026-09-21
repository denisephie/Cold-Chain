import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------------------------------------#
# Data Loading & Feature Engineering
dataset_1 = pd.read_csv("./shipment-sensor-dataset.csv")

numerical_features = [
    "transit_days",
    "door_opens",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "temp_std_c",
    "temp_recovery_rate",
    "rh_mean",
    "rh_std",
    "rh_max",
    "product_volume_l",
    "fill_ratio",
    "leg_count",
    "sensor_gap_hours",
    "vibration_index",
]
categorical_features = ["package_type", "carrier_id", "origin_zone", "dest_zone"]
target = "silent_failure"

engineered_dataset = dataset_1.copy()

engineered_dataset["temp_x_rh_stress"] = (
    engineered_dataset["temp_max_c"] * engineered_dataset["rh_std"]
)
engineered_dataset["opens_x_transit"] = (
    engineered_dataset["door_opens"] * engineered_dataset["transit_days"]
)
engineered_dataset["temp_breach_ratio"] = engineered_dataset[
    "temp_max_c"
] / engineered_dataset["temp_mean_c"].abs().replace(0, np.nan)
engineered_dataset["recovery_x_std"] = (
    engineered_dataset["temp_recovery_rate"] * engineered_dataset["temp_std_c"]
)
engineered_dataset["temp_range_c"] = (
    engineered_dataset["temp_max_c"] - engineered_dataset["temp_min_c"]
)
engineered_dataset["door_opens_per_day"] = engineered_dataset[
    "door_opens"
] / engineered_dataset["transit_days"].replace(0, np.nan)
engineered_dataset["temp_variability_ratio"] = engineered_dataset[
    "temp_std_c"
] / engineered_dataset["temp_mean_c"].abs().replace(0, np.nan)
engineered_dataset["humidity_variability"] = (
    engineered_dataset["rh_max"] - engineered_dataset["rh_mean"]
)
engineered_dataset["vibration_per_day"] = engineered_dataset[
    "vibration_index"
] / engineered_dataset["transit_days"].replace(0, np.nan)
engineered_dataset["legs_per_day"] = engineered_dataset[
    "leg_count"
] / engineered_dataset["transit_days"].replace(0, np.nan)

engineered_dataset = engineered_dataset.replace([np.inf, -np.inf], np.nan)

engineered_features = [
    "temp_x_rh_stress",
    "opens_x_transit",
    "temp_breach_ratio",
    "recovery_x_std",
    "temp_range_c",
    "door_opens_per_day",
    "temp_variability_ratio",
    "humidity_variability",
    "vibration_per_day",
    "legs_per_day",
]

final_features = numerical_features + engineered_features + categorical_features

# --------------------------------------------------------------------------------------------------#
# Encoding & Train/Test Split
model_dataset = engineered_dataset[final_features + [target]].copy()
model_dataset = model_dataset.fillna(model_dataset.median(numeric_only=True))

encoded_dataset = model_dataset.copy()
label_encoders = {}

for feature in categorical_features:
  encoder = LabelEncoder()
  encoded_dataset[feature] = encoder.fit_transform(
      encoded_dataset[feature].astype(str)
  )
  label_encoders[feature] = encoder

X = encoded_dataset[final_features]
y = encoded_dataset[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=67, stratify=y
)

# Calculate scale_pos_weight correctly for un-sampled data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")

# --------------------------------------------------------------------------------------------------#
# Cross-Validation & Hyperparameter Tuning (Without SMOTE)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=67)

param_dist = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0, 0.1, 0.2, 0.3, 0.5],
    "reg_alpha": [0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 1.5, 2.0],
}

tuning_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=67,
)

search = RandomizedSearchCV(
    tuning_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring="f1",
    cv=cv,
    random_state=67,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

# --------------------------------------------------------------------------------------------------#
# Training & Evaluation
model = search.best_estimator_
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation at Default Threshold (0.50) ---")
print(classification_report(y_test, y_pred))

# Threshold Search
thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []

for t in thresholds:
  y_p = (y_prob >= t).astype(int)
  threshold_results.append({
      "threshold": round(t, 2),
      "precision": precision_score(y_test, y_p, zero_division=0),
      "recall": recall_score(y_test, y_p, zero_division=0),
      "f1": f1_score(y_test, y_p, zero_division=0),
  })

threshold_df = pd.DataFrame(threshold_results)
best_f1_row = threshold_df.loc[threshold_df["f1"].idxmax()]
best_threshold = best_f1_row["threshold"]

print(f"\n--- Model Evaluation at Optimal Threshold ({best_threshold}) ---")
y_pred_best = (y_prob >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_best))

# --------------------------------------------------------------------------------------------------#
# Save Artifacts
joblib.dump(model, "./cold_chain_xgboost.joblib")
joblib.dump(label_encoders, "./label_encoders.joblib")
joblib.dump(final_features, "./model_features.joblib")

test_results = X_test.copy()
test_results["actual"] = y_test.values
test_results["predicted_best_threshold"] = y_pred_best
test_results["failure_probability"] = y_prob
test_results.to_csv("./test_predictions.csv", index=False)

print("\nArtifacts successfully saved.")