import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
    )

dataset_1 = pd.read_csv("./shipment-sensor-dataset.csv")

# Baseline
baseline_features = [
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

X = dataset_1[baseline_features]
y = dataset_1["silent_failure"]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=67, stratify=y )

# Baseline XGBoost
baseline_model = xgb.XGBClassifier( n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss", random_state=67 )

baseline_model.fit( X_train, y_train)


# Baseline Evaluation
baseline_pred = baseline_model.predict(X_test)
baseline_prob = baseline_model.predict_proba(X_test)[:, 1]
baseline_accuracy = accuracy_score(y_test, baseline_pred)
baseline_precision = precision_score(y_test, baseline_pred)
baseline_recall = recall_score( y_test, baseline_pred)
baseline_f1 = f1_score(y_test, baseline_pred)
baseline_auc = roc_auc_score(y_test, baseline_prob)

print("Baseline XGBoost Evaluation")
print(f"Accuracy:  {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall:    {baseline_recall:.4f}")
print(f"F1 Score:  {baseline_f1:.4f}")
print(f"ROC-AUC:   {baseline_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test,baseline_pred))

# Baseline Conf Matrix
conf_matrix = confusion_matrix(y_test, baseline_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
display.plot()
plt.title("Baseline XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# Baseline Feature Importance
baseline_importance = pd.DataFrame({"feature": baseline_features, "importance": baseline_model.feature_importances_})
baseline_importance = baseline_importance.sort_values(by="importance", ascending=False)
print("\nBaseline Feature Importance:")
print(baseline_importance)

plt.figure(figsize=(10, 7))
plt.barh(baseline_importance["feature"], baseline_importance["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Baseline XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Baseline Correlation 
print("\nCorrelation With Silent Failure:")
correlation = dataset_1[baseline_features + ["silent_failure"]].corr()["silent_failure"].sort_values(ascending=False)
print(correlation)

#----------------------------------------------------------------------------------------------------------------------------------#
# Categorical
categorical_features = ["package_type", "carrier_id", "origin_zone", "dest_zone"]
encoded_dataset = dataset_1.copy()
label_encoders = {}
for feature in categorical_features:
    encoder = LabelEncoder()
    encoded_dataset[feature] = encoder.fit_transform(encoded_dataset[feature].astype(str))
    label_encoders[feature] = encoder

categorical_model_features = baseline_features + categorical_features
X_categorical = encoded_dataset[categorical_model_features]
y_categorical = encoded_dataset["silent_failure"]
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_categorical, y_categorical, test_size=0.20, random_state=67,stratify=y_categorical )

# Categorical XGBoost
categorical_model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, 
    subsample=0.8, colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",random_state=67)

categorical_model.fit(X_train_cat, y_train_cat)


# Categorical Evaluation
categorical_pred = categorical_model.predict(X_test_cat)
categorical_prob = categorical_model.predict_proba(X_test_cat)[:, 1]
categorical_accuracy = accuracy_score(y_test_cat, categorical_pred)
categorical_precision = precision_score(y_test_cat, categorical_pred)
categorical_recall = recall_score(y_test_cat, categorical_pred)
categorical_f1 = f1_score(y_test_cat, categorical_pred)
categorical_auc = roc_auc_score(y_test_cat, categorical_prob)


print("Categorical XGBoost Evaluation")
print(f"Accuracy:  {categorical_accuracy:.4f}")
print(f"Precision: {categorical_precision:.4f}")
print(f"Recall:    {categorical_recall:.4f}")
print(f"F1 Score:  {categorical_f1:.4f}")
print(f"ROC-AUC:   {categorical_auc:.4f}")


# Categorical Feature Importance
categorical_importance = pd.DataFrame({"feature": categorical_model_features,"importance": categorical_model.feature_importances_})
categorical_importance = categorical_importance.sort_values(by="importance", ascending=False)
print("\nFeature Importance With Categorical Features:")
print(categorical_importance)

plt.figure(figsize=(10, 8))
plt.barh(categorical_importance["feature"], categorical_importance["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance With Categorical Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------#
# Feature Engineering
engineered_dataset = dataset_1.copy()

engineered_dataset["temp_range_c"] = engineered_dataset["temp_max_c"] - engineered_dataset["temp_min_c"]
engineered_dataset["door_opens_per_day"] = engineered_dataset["door_opens"] / engineered_dataset["transit_days"].replace(0, np.nan)
engineered_dataset["effective_volume_l"] = engineered_dataset["product_volume_l"] * engineered_dataset["fill_ratio"]
engineered_dataset["temp_variability_ratio"] = engineered_dataset["temp_std_c"] / engineered_dataset["temp_mean_c"].abs().replace(0, np.nan)
engineered_dataset["humidity_variability"] = engineered_dataset["rh_max"] - engineered_dataset["rh_mean"]
engineered_dataset["vibration_per_day"] = engineered_dataset["vibration_index"] / engineered_dataset["transit_days"].replace(0, np.nan)
engineered_dataset["legs_per_day"] = engineered_dataset["leg_count"] / engineered_dataset["transit_days"].replace(0, np.nan)

engineered_dataset = engineered_dataset.replace([np.inf, -np.inf], np.nan)

engineered_features = ["temp_range_c", "door_opens_per_day", "effective_volume_l", 
    "temp_variability_ratio", "humidity_variability", "vibration_per_day", "legs_per_day"]

print("\nEngineered Features:")
print(engineered_features)

# Engineered Correlation
print("\nEngineered Feature Correlation:")
engineered_correlation = engineered_dataset[engineered_features + ["silent_failure"]].corr()["silent_failure"].sort_values(ascending=False)
print(engineered_correlation)

# Final Feature Set
final_features = baseline_features + engineered_features
X_engineered = engineered_dataset[final_features]
y_engineered = engineered_dataset["silent_failure"]
X_engineered = X_engineered.fillna(X_engineered.median(numeric_only=True))

X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split( X_engineered, y_engineered, test_size=0.20, random_state=67, stratify=y_engineered )

# Engineered XGBoost
engineered_model = xgb.XGBClassifier( n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss", random_state=67 )

engineered_model.fit( X_train_eng, y_train_eng )

# Engineered Evaluation
engineered_pred = engineered_model.predict(X_test_eng)
engineered_prob = engineered_model.predict_proba(X_test_eng)[:, 1]
engineered_accuracy = accuracy_score(y_test_eng, engineered_pred)
engineered_precision = precision_score(y_test_eng, engineered_pred)
engineered_recall = recall_score(y_test_eng, engineered_pred)
engineered_f1 = f1_score(y_test_eng, engineered_pred)
engineered_auc = roc_auc_score(y_test_eng, engineered_prob)

print("Engineered XGBoost Evaluation")
print(f"Accuracy:  {engineered_accuracy:.4f}")
print(f"Precision: {engineered_precision:.4f}")
print(f"Recall:    {engineered_recall:.4f}")
print(f"F1 Score:  {engineered_f1:.4f}")
print(f"ROC-AUC:   {engineered_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_eng, engineered_pred))

# Engineered Conf Matrix
engineered_conf_matrix = confusion_matrix(y_test_eng, engineered_pred)
print("\nConfusion Matrix:")
print(engineered_conf_matrix)

display = ConfusionMatrixDisplay(confusion_matrix=engineered_conf_matrix)
display.plot()
plt.title("Engineered XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# Engineered Feature Importance
engineered_importance = pd.DataFrame({"feature": final_features, "importance": engineered_model.feature_importances_})
engineered_importance = engineered_importance.sort_values(by="importance", ascending=False)
print("\nEngineered Model Feature Importance:")
print(engineered_importance)

plt.figure(figsize=(10, 10))
plt.barh(engineered_importance["feature"], engineered_importance["importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance - Engineered Model")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------------------------------------#
# Model Comparison
comparison = pd.DataFrame({
    "Model": ["Baseline XGBoost", "XGBoost + Categorical", "XGBoost + Engineered"],
    "Accuracy": [baseline_accuracy, categorical_accuracy, engineered_accuracy],
    "Precision": [baseline_precision, categorical_precision, engineered_precision],
    "Recall": [baseline_recall, categorical_recall, engineered_recall],
    "F1": [baseline_f1, categorical_f1, engineered_f1],
    "ROC-AUC": [baseline_auc, categorical_auc, engineered_auc]
})
print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

# ROC Curves
baseline_fpr, baseline_tpr, _ = roc_curve(y_test, baseline_prob)
categorical_fpr, categorical_tpr, _ = roc_curve(y_test_cat, categorical_prob)
engineered_fpr, engineered_tpr, _ = roc_curve(y_test_eng, engineered_prob)

plt.figure(figsize=(8, 6))
plt.plot(baseline_fpr, baseline_tpr, label=f"Baseline (AUC = {baseline_auc:.3f})")
plt.plot(categorical_fpr, categorical_tpr, label=f"Categorical (AUC = {categorical_auc:.3f})")
plt.plot(engineered_fpr, engineered_tpr, label=f"Engineered (AUC = {engineered_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show()