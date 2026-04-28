"""End-to-end training pipeline for the fraud-detection Random Forest.

This script performs the full ML workflow:
    1. Exploratory Data Analysis (EDA) — class balance, distributions,
       correlation heatmap, box-plots, and statistical tests.
    2. Feature engineering — RobustScaler on Amount and Time.
    3. Resampling — SMOTE on the training split to address class
       imbalance.
    4. Model training — ``RandomForestClassifier`` with tuned
       hyperparameters.
    5. Evaluation — confusion matrix, classification report, ROC-AUC.

All EDA plots are saved to ``<project_root>/eda_output/``.

Pipeline step: Step 3 – Model training.

Usage:
    python training/train.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np  # noqa: F401 (used implicitly by downstream code)
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
output_dir = os.path.join(project_root, "eda_output")
os.makedirs(output_dir, exist_ok=True)

data_path = os.path.join(project_root, "data", "creditcard.csv")

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------

print(f"Loading data from {data_path}...")
try:
    df = pd.read_csv(data_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Could not find {data_path}.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# 2. Target variable analysis (class imbalance)
# ---------------------------------------------------------------------------

print("-" * 50)
print("1. Target Variable Analysis (Class Imbalance)")
print("-" * 50)

class_counts = df["Class"].value_counts()
print(class_counts)
print(f"Fraud Percentage: {class_counts[1] / len(df) * 100:.3f}%")

plt.figure(figsize=(8, 6))
sns.countplot(x="Class", data=df)
plt.title("Class Distribution (0: Normal, 1: Fraud)")
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

# ---------------------------------------------------------------------------
# 3. Univariate analysis (distributions)
# ---------------------------------------------------------------------------

print("\n" + "-" * 50)
print("2. Univariate Analysis (Distribution)")
print("-" * 50)

fig, axes = plt.subplots(1, 2, figsize=(18, 4))
sns.histplot(df["Amount"], bins=40, ax=axes[0], color="r", kde=True)
axes[0].set_title("Distribution of Transaction Amount")
axes[0].set_xlim([min(df["Amount"]), max(df["Amount"])])

sns.histplot(df["Time"], bins=40, ax=axes[1], color="b", kde=True)
axes[1].set_title("Distribution of Transaction Time")
axes[1].set_xlim([min(df["Time"]), max(df["Time"])])

plt.savefig(os.path.join(output_dir, "amount_time_distribution.png"))
plt.close()

# Compare selected feature distributions between fraud and normal classes.
features_to_plot = ["V14", "V17", "V12", "V10"]

for feature in features_to_plot:
    plt.figure(figsize=(8, 6))
    sns.kdeplot(df[df["Class"] == 0][feature], label="Class 0 (Normal)", fill=True)
    sns.kdeplot(df[df["Class"] == 1][feature], label="Class 1 (Fraud)", fill=True)
    plt.title(f"Distribution of {feature} by Class")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{feature}_distribution.png"))
    plt.close()

# ---------------------------------------------------------------------------
# 4. Correlation analysis (balanced sub-sample)
# ---------------------------------------------------------------------------

print("\n" + "-" * 50)
print("3. Correlation Analysis (Pearson & Sub-sampling)")
print("-" * 50)

# Random undersampling to build a balanced sub-sample for correlation.
fraud_df = df.loc[df["Class"] == 1]
non_fraud_df = df.loc[df["Class"] == 0].sample(n=len(fraud_df), random_state=42)
normalized_df = pd.concat([fraud_df, non_fraud_df])
new_df = normalized_df.sample(frac=1, random_state=42)  # Shuffle

sub_sample_corr = new_df.corr()
plt.figure(figsize=(24, 20))
sns.heatmap(sub_sample_corr, cmap="coolwarm_r", annot_kws={"size": 20})
plt.title("SubSample Correlation Matrix (Balanced Dataset)", fontsize=14)
plt.savefig(os.path.join(output_dir, "correlation_matrix_subsample.png"))
plt.close()

print("Key Negative Correlations (lower value → more likely fraud):")
print(sub_sample_corr["Class"].sort_values()[:5])
print("\nKey Positive Correlations (higher value → more likely fraud):")
print(sub_sample_corr["Class"].sort_values(ascending=False)[1:6])

# ---------------------------------------------------------------------------
# 5. Outlier analysis (box-plots)
# ---------------------------------------------------------------------------

print("\n" + "-" * 50)
print("4. Outlier Analysis (Boxplots)")
print("-" * 50)

fig, axes = plt.subplots(ncols=4, figsize=(20, 6))
colors = ["#0101DF", "#DF0101"]

sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title("V17 vs Class – Negative Correlation")

sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title("V14 vs Class – Negative Correlation")

sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title("V12 vs Class – Negative Correlation")

sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title("V10 vs Class – Negative Correlation")

plt.savefig(os.path.join(output_dir, "boxplots_negative_corr.png"))
plt.close()

# ---------------------------------------------------------------------------
# 6. Statistical testing (T-test on V14)
# ---------------------------------------------------------------------------

print("\n" + "-" * 50)
print("5. Statistical Testing (T-test / Mann-Whitney)")
print("-" * 50)

v14_fraud = new_df["V14"].loc[new_df["Class"] == 1].values
v14_normal = new_df["V14"].loc[new_df["Class"] == 0].values

t_stat, p_val = stats.ttest_ind(v14_fraud, v14_normal)
print(f"T-test on V14 (Subsample): t-statistic = {t_stat:.4f}, p-value = {p_val:.4e}")
if p_val < 0.05:
    print("Conclusion: The difference in V14 means between Fraud and Normal is statistically significant.")
else:
    print("Conclusion: The difference in V14 means is not statistically significant.")

# ---------------------------------------------------------------------------
# 7. Feature engineering – scaling
# ---------------------------------------------------------------------------

print("\n" + "-" * 50)
print("6. Data Cleaning, Scaling, and Pre-processing")
print("-" * 50)
print("Applying RobustScaler to 'Amount' and 'Time'...")

rob_scaler = RobustScaler()

df["scaled_amount"] = rob_scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["scaled_time"] = rob_scaler.fit_transform(df["Time"].values.reshape(-1, 1))

df.drop(["Time", "Amount"], axis=1, inplace=True)

# Move scaled columns to the front of the DataFrame.
scaled_amount = df["scaled_amount"]
scaled_time = df["scaled_time"]

df.drop(["scaled_amount", "scaled_time"], axis=1, inplace=True)
df.insert(0, "scaled_amount", scaled_amount)
df.insert(1, "scaled_time", scaled_time)
print("Scaling complete. Columns re-ordered.")

# ---------------------------------------------------------------------------
# 8. Train / test split + SMOTE resampling
# ---------------------------------------------------------------------------

print("\nPreparing for Modelling: train/test split then SMOTE on train only.")
X = df.drop("Class", axis=1)
y = df["Class"]

# Split *before* SMOTE to prevent data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"Original X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

print("\nApplying SMOTE to X_train...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"SMOTE applied. X_train_sm shape: {X_train_sm.shape}, y_train_sm shape: {y_train_sm.shape}")
print(f"Class distribution after SMOTE:\n{y_train_sm.value_counts()}")

print("\nEDA and feature-engineering pipeline complete. Data is ready for training.")

# ---------------------------------------------------------------------------
# 9. Random Forest training
# ---------------------------------------------------------------------------

print("Training Random Forest (this may take a few minutes)...\n")

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=40,
    min_samples_split=35,
    min_samples_leaf=15,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
rf_clf.fit(X_train_sm, y_train_sm)

# ---------------------------------------------------------------------------
# 10. Evaluation on test set
# ---------------------------------------------------------------------------

print("Evaluating Random Forest on Test Set...")
y_pred_rf = rf_clf.predict(X_test)
y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_rf):.4f}")
