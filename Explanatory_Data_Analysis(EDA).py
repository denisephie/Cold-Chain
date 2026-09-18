import pandas as pd
import matplotlib.pyplot as plt

dataset_1 = pd.read_csv("./shipment-sensor-dataset.csv")
print("Dataset:")
print(dataset_1)
print("\nDataset Information:")
print(dataset_1.info())
print("\nStatistical Summary:")
print(dataset_1.describe())
print("\nMissing Values:")
print(dataset_1.isnull().sum())

numerical_features = [
    "transit_days", "door_opens", "temp_mean_c", "temp_max_c", "temp_min_c",
    "temp_std_c", "temp_recovery_rate", "rh_mean", "rh_std", "rh_max",
    "product_volume_l", "fill_ratio", "leg_count", "sensor_gap_hours", "vibration_index"
]
categorical_features = ["package_type", "carrier_id", "origin_zone", "dest_zone"]


# Numerical Feature Distribution - Part 1
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(numerical_features[:8]):
    axes[i].hist(dataset_1[feature].dropna(), bins=30)
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")

for j in range(len(numerical_features[:8]), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Numerical Feature Distributions (Part 1)", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# Numerical Feature Distribution - Part 2
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(numerical_features[8:]):
    axes[i].hist(dataset_1[feature].dropna(), bins=30)
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")

for j in range(len(numerical_features[8:]), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Numerical Feature Distributions (Part 2)", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# Numerical Features by Silent Failure - Part 1
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(numerical_features[:8]):
    dataset_1[dataset_1["silent_failure"] == 0][feature].plot(
        kind="hist", bins=30, alpha=0.5, label="No Failure", ax=axes[i] )

    dataset_1[dataset_1["silent_failure"] == 1][feature].plot(
        kind="hist", bins=30, alpha=0.5, label="Failure", ax=axes[i] )

    axes[i].set_title(f"{feature} by Silent Failure")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")
    axes[i].legend()

for j in range(len(numerical_features[:8]), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Numerical Features by Silent Failure (Part 1)", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()


# Numerical Features by Silent Failure - Part 2
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(numerical_features[8:]):
    dataset_1[dataset_1["silent_failure"] == 0][feature].plot(
        kind="hist", bins=30, alpha=0.5, label="No Failure", ax=axes[i] )
    
    dataset_1[dataset_1["silent_failure"] == 1][feature].plot(
        kind="hist", bins=30, alpha=0.5, label="Failure", ax=axes[i] )
    
    axes[i].set_title(f"{feature} by Silent Failure")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")
    axes[i].legend()

for j in range(len(numerical_features[8:]), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Numerical Features by Silent Failure (Part 2)", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# Categorical Features by Silent Failure
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    dataset_1.groupby([feature, "silent_failure"]).size().unstack(fill_value=0).plot(
        kind="bar", ax=axes[i], alpha=0.8 )

    axes[i].set_title(f"{feature} by Silent Failure")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")
    axes[i].tick_params(axis="x", rotation=45)
    axes[i].legend(["No Failure", "Failure"])

plt.suptitle("Categorical Features by Silent Failure", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()