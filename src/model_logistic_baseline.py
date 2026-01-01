import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)

# ==========================================
# 1. SETUP & DIRECTORIES
# ==========================================
# Create a folder to save plots if it doesn't exist
PLOT_DIR = "plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    print(f"Created directory: {PLOT_DIR}")

# ==========================================
# 2. DATA LOADING
# ==========================================
# Note: Using .xlsx as in your Data_preparation.py
try:
    train_df = pd.read_excel("train_data_final.xlsx")
    test_df = pd.read_excel("test_data_final.xlsx")
except FileNotFoundError:
    print("Excel files not found. Ensure Data_preparation.py was run successfully.")
    # Fallback to CSV if needed (for internal testing)
    train_df = pd.read_csv("train_data_final.csv")
    test_df = pd.read_csv("test_data_final.csv")

X_train = train_df.drop(columns=["OUTCOME"])
y_train = train_df["OUTCOME"].astype(int)

X_test = test_df.drop(columns=["OUTCOME"])
y_test = test_df["OUTCOME"].astype(int)

# Ensure boolean columns are converted to int for Logistic Regression
for df in [X_train, X_test]:
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

print(f"Data Loaded: Train {X_train.shape}, Test {X_test.shape}")

# ==========================================
# 3. HYPERPARAMETER TUNING (Grid Search)
# ==========================================
print("\n--- Starting Hyperparameter Tuning (Grid Search) ---")

# Define the model
base_log_reg = LogisticRegression(max_iter=5000, solver='liblinear')

# Define the grid of parameters to search
# C: Inverse of regularization strength (smaller = stronger regularization)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}

# Setup Stratified K-Fold
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search with the metrics we care about
grid_search = GridSearchCV(
    estimator=base_log_reg,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1',  # Optimizing for F1-Score as requested
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best Parameters Found: {grid_search.best_params_}")

# ==========================================
# 4. CROSS-VALIDATION PERFORMANCE (K-Fold Results)
# ==========================================
cv_results = pd.DataFrame(grid_search.cv_results_)
best_idx = grid_search.best_index_

print("\n--- Cross-Validation Results (Best Fold Average) ---")
print(f"Mean F1-Score: {cv_results.loc[best_idx, 'mean_test_score']:.4f}")
print(f"Std F1-Score:  {cv_results.loc[best_idx, 'std_test_score']:.4f}")

# ==========================================
# 5. FINAL EVALUATION ON TEST SET (Hold-out)
# ==========================================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1] # These are the Posteriors P(C=1|x)

# Calculate final metrics
p = precision_score(y_test, y_pred)
r = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n--- FINAL EVALUATION (Logistic Regression - Test Set) ---")
print(f"Precision : {p:.4f}")
print(f"Recall    : {r:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 6. VISUALIZATION MODULE
# ==========================================

# 6.1 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix: Logistic Regression (Baseline)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/logistic_cm.png")
plt.show()

# 6.2 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title('Receiver Operating Characteristic (ROC) - Logistic')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(f"{PLOT_DIR}/logistic_roc.png")
plt.show()

# 6.3 Precision-Recall Curve (Crucial for Insurance/Imbalanced data)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(rec_curve, prec_curve, color='green', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Logistic')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.savefig(f"{PLOT_DIR}/logistic_pr_curve.png")
plt.show()

# 6.4 Feature Importance (Coefficients for Logistic)
coefs = pd.Series(best_model.coef_[0], index=X_train.columns)
top_coefs = coefs.sort_values(ascending=False).head(10) # Showing top 10 impact variables

plt.figure(figsize=(10, 6))
top_coefs.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Top 10 Feature Coefficients (Impact on Claim)")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45, ha='right')
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/logistic_feature_impact.png")
plt.show()

print(f"\nAll plots saved in the '{PLOT_DIR}' directory.")