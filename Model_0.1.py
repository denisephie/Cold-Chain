import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

dataset_1 = pd.read_csv("./shipment-sensor-dataset.csv")

print("Dataset Shape:")
print(dataset_1.shape)


# ============================================================
# 2. DEFINE FEATURES
# ============================================================

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


features = numerical_features + categorical_features

target = "silent_failure"


print("\nFeatures:")
for feature in features:
    print(feature)


# ============================================================
# 3. CREATE DATASET FOR MODEL
# ============================================================

model_dataset = dataset_1[
    features + [target]
].copy()


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:")

print(
    model_dataset.isnull().sum()
)


# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

encoded_dataset = model_dataset.copy()

label_encoders = {}

for feature in categorical_features:

    encoder = LabelEncoder()

    encoded_dataset[feature] = encoder.fit_transform(
        encoded_dataset[feature].astype(str)
    )

    label_encoders[feature] = encoder

    print(f"\nEncoding for {feature}:")

    for index, category in enumerate(encoder.classes_):
        print(f"{category} -> {index}")


# ============================================================
# 6. SEPARATE X AND Y
# ============================================================

X = encoded_dataset[
    features
]

y = encoded_dataset[
    target
]


print("\nX Shape:")
print(X.shape)

print("\ny Shape:")
print(y.shape)


# ============================================================
# 7. TRAIN / TEST SPLIT
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
# 8. CREATE XGBOOST MODEL
# ============================================================

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


print("\nModel Training Complete")


# ============================================================
# 10. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

y_prob = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 11. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


print("\n================================")
print("XGBOOST CATEGORICAL MODEL")
print("================================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "XGBoost Categorical Model - Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})


feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)


print("\nFeature Importance:")

print(
    feature_importance.to_string(
        index=False
    )
)


plt.figure(
    figsize=(10, 8)
)

plt.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "XGBoost Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()


# ============================================================
# 15. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "./cold_chain_xgboost.joblib"
)

print(
    "\nModel saved to:"
)

print(
    "./cold_chain_xgboost.joblib"
)


# ============================================================
# 16. SAVE LABEL ENCODERS
# ============================================================

joblib.dump(
    label_encoders,
    "./label_encoders.joblib"
)

print(
    "\nLabel encoders saved to:"
)

print(
    "./label_encoders.joblib"
)


# ============================================================
# 17. SAVE FEATURE LIST
# ============================================================

joblib.dump(
    features,
    "./model_features.joblib"
)

print(
    "\nFeature list saved to:"
)

print(
    "./model_features.joblib"
)


# ============================================================
# 18. SAVE TEST PREDICTIONS
# ============================================================

test_results = X_test.copy()

test_results["actual"] = y_test.values

test_results["predicted"] = y_pred

test_results["failure_probability"] = y_prob


test_results.to_csv(
    "./test_predictions.csv",
    index=False
)


print(
    "\nTest predictions saved to:"
)

print(
    "./test_predictions.csv"
)


# ============================================================
# 19. SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "model": "XGBoost",
    "model_type": "Binary Classification",
    "target": target,
    "features": features,
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "random_state": 42,
    "test_size": 0.20
}


joblib.dump(
    model_info,
    "./model_info.joblib"
)


print(
    "\nModel information saved to:"
)

print(
    "./model_info.joblib"
)

print("FILES CREATED")