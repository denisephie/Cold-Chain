import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import numpy as np
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)


dataset_1 = pd.read_csv("./shipment-sensor-dataset.csv")

print("Dataset Shape:")
print(dataset_1.shape)


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
    "vibration_index"
]

categorical_features = [
    "package_type",
    "carrier_id",
    "origin_zone",
    "dest_zone"
]

target = "silent_failure"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

engineered_dataset = dataset_1.copy()

engineered_dataset["temp_x_rh_stress"]        = engineered_dataset["temp_max_c"] * engineered_dataset["rh_std"]
engineered_dataset["opens_x_transit"]          = engineered_dataset["door_opens"] * engineered_dataset["transit_days"]
engineered_dataset["temp_breach_ratio"]        = engineered_dataset["temp_max_c"] / engineered_dataset["temp_mean_c"].abs().replace(0, np.nan)
engineered_dataset["recovery_x_std"]           = engineered_dataset["temp_recovery_rate"] * engineered_dataset["temp_std_c"]
engineered_dataset["temp_range_c"]             = engineered_dataset["temp_max_c"] - engineered_dataset["temp_min_c"]
engineered_dataset["door_opens_per_day"]       = engineered_dataset["door_opens"] / engineered_dataset["transit_days"].replace(0, np.nan)
engineered_dataset["temp_variability_ratio"]   = engineered_dataset["temp_std_c"] / engineered_dataset["temp_mean_c"].abs().replace(0, np.nan)
engineered_dataset["humidity_variability"]     = engineered_dataset["rh_max"] - engineered_dataset["rh_mean"]
engineered_dataset["vibration_per_day"]        = engineered_dataset["vibration_index"] / engineered_dataset["transit_days"].replace(0, np.nan)
engineered_dataset["legs_per_day"]             = engineered_dataset["leg_count"] / engineered_dataset["transit_days"].replace(0, np.nan)

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
    "legs_per_day"
]

final_features = numerical_features + engineered_features + categorical_features

print("\nFeatures:")
for feature in final_features:
    print(feature)


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

model_dataset = engineered_dataset[final_features + [target]].copy()
model_dataset = model_dataset.fillna(model_dataset.median(numeric_only=True))

print("\nMissing Values:")
print(model_dataset.isnull().sum())

encoded_dataset = model_dataset.copy()
label_encoders = {}

for feature in categorical_features:
    encoder = LabelEncoder()
    encoded_dataset[feature] = encoder.fit_transform(encoded_dataset[feature].astype(str))
    label_encoders[feature] = encoder

    print(f"\nEncoding for {feature}:")
    for index, category in enumerate(encoder.classes_):
        print(f"{category} -> {index}")


# ============================================================
# SEPARATE X AND Y
# ============================================================

X = encoded_dataset[final_features]
y = encoded_dataset[target]

print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


# ============================================================
# CLASS IMBALANCE
# ============================================================

print("\nClass Distribution (Train):")
print(y_train.value_counts())

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

print("\nscale_pos_weight:")
print(round(scale_pos_weight, 4))

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nClass Distribution After SMOTE:")
print(pd.Series(y_train_resampled).value_counts())


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

cv_results = cross_validate(
    cv_model,
    X_train_resampled,
    y_train_resampled,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    return_train_score=False
)

print("\n5-Fold Cross-Validation")
print(f"Accuracy:  {cv_results['test_accuracy'].mean():.4f} +/- {cv_results['test_accuracy'].std():.4f}")
print(f"Precision: {cv_results['test_precision'].mean():.4f} +/- {cv_results['test_precision'].std():.4f}")
print(f"Recall:    {cv_results['test_recall'].mean():.4f} +/- {cv_results['test_recall'].std():.4f}")
print(f"F1 Score:  {cv_results['test_f1'].mean():.4f} +/- {cv_results['test_f1'].std():.4f}")
print(f"ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} +/- {cv_results['test_roc_auc'].std():.4f}")


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

param_dist = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 5, 6, 7, 8],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8],
    "min_child_weight": [1, 3, 5, 7],
    "gamma":            [0, 0.1, 0.2, 0.3, 0.5],
    "reg_alpha":        [0, 0.1, 0.5, 1.0],
    "reg_lambda":       [0.5, 1.0, 1.5, 2.0]
}

tuning_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

search = RandomizedSearchCV(
    tuning_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring="f1",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train_resampled, y_train_resampled)

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest CV F1:")
print(round(search.best_score_, 4))


# ============================================================
# TRAIN TUNED MODEL
# ============================================================

model = search.best_estimator_

model.fit(X_train_resampled, y_train_resampled)

print("\nModel Training Complete")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)

print("\n================================")
print("TUNED XGBOOST MODEL")
print("================================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Tuned XGBoost - Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": final_features,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(by="importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance.to_string(index=False))

plt.figure(figsize=(12, 10))
plt.barh(feature_importance["feature"], feature_importance["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Tuned XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []

for threshold in thresholds:
    y_pred_thresh = (y_prob >= threshold).astype(int)
    threshold_results.append({
        "threshold": round(threshold, 2),
        "accuracy":  accuracy_score(y_test, y_pred_thresh),
        "precision": precision_score(y_test, y_pred_thresh, zero_division=0),
        "recall":    recall_score(y_test, y_pred_thresh, zero_division=0),
        "f1":        f1_score(y_test, y_pred_thresh, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nThreshold Analysis:")
print(threshold_df.to_string(index=False))

best_f1_row = threshold_df.loc[threshold_df["f1"].idxmax()]

print("\nBest Threshold by F1:")
print(f"{best_f1_row['threshold']} (F1 = {best_f1_row['f1']:.4f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", marker="o")
axes[0].plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", marker="o")
axes[0].plot(threshold_df["threshold"], threshold_df["f1"], label="F1 Score", marker="o")
axes[0].axvline(x=best_f1_row["threshold"], linestyle="--", color="gray", label=f"Best F1 Threshold ({best_f1_row['threshold']})")
axes[0].set_title("Precision / Recall / F1 vs Threshold")
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Score")
axes[0].legend()

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
axes[1].plot([0, 1], [0, 1], linestyle="--")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()

plt.suptitle("Threshold Analysis", fontsize=14)
plt.tight_layout()
plt.show()

best_threshold = best_f1_row["threshold"]
y_pred_best = (y_prob >= best_threshold).astype(int)

print(f"\nEvaluation at Best Threshold ({best_threshold}):")
print(classification_report(y_test, y_pred_best))


# ============================================================
# SHAP
# ============================================================

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("\nSHAP Analysis")

shap.summary_plot(shap_values, X_test, feature_names=final_features, show=False)
plt.title("SHAP Summary Plot")
plt.tight_layout()
plt.show()

shap.summary_plot(shap_values, X_test, feature_names=final_features, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()

shap_importance = pd.DataFrame({
    "feature": final_features,
    "shap_importance": np.abs(shap_values).mean(axis=0)
})

shap_importance = shap_importance.sort_values(by="shap_importance", ascending=False)

print("\nSHAP Feature Importance:")
print(shap_importance.to_string(index=False))


# ============================================================
# SAVE ARTIFACTS
# ============================================================

joblib.dump(model, "./cold_chain_xgboost.joblib")
print("\nModel saved to:")
print("./cold_chain_xgboost.joblib")

joblib.dump(label_encoders, "./label_encoders.joblib")
print("\nLabel encoders saved to:")
print("./label_encoders.joblib")

joblib.dump(final_features, "./model_features.joblib")
print("\nFeature list saved to:")
print("./model_features.joblib")

test_results = X_test.copy()
test_results["actual"]                   = y_test.values
test_results["predicted"]                = y_pred
test_results["predicted_best_threshold"] = y_pred_best
test_results["failure_probability"]      = y_prob

test_results.to_csv("./test_predictions.csv", index=False)
print("\nTest predictions saved to:")
print("./test_predictions.csv")

model_info = {
    "model": "XGBoost",
    "model_type": "Binary Classification",
    "target": target,
    "features": final_features,
    "numerical_features": numerical_features,
    "engineered_features": engineered_features,
    "categorical_features": categorical_features,
    "scale_pos_weight": scale_pos_weight,
    "best_params": search.best_params_,
    "best_threshold": best_threshold,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "random_state": 42,
    "test_size": 0.20
}

joblib.dump(model_info, "./model_info.joblib")
print("\nModel information saved to:")
print("./model_info.joblib")

print("\nFILES CREATED")