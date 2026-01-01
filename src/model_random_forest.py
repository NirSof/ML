import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, classification_report
)

# ==========================================
# 1. SETUP & DIRECTORIES
# ==========================================
PLOT_DIR = "plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

# ==========================================
# 2. DATA LOADING
# ==========================================
print("--- Loading Prepared Data ---")
try:
    train_df = pd.read_excel("train_data_final.xlsx")
    test_df = pd.read_excel("test_data_final.xlsx")
except Exception:
    # Fallback if only CSVs are available
    train_df = pd.read_csv("train_data_final.csv")
    test_df = pd.read_csv("test_data_final.csv")

X_train = train_df.drop(columns=["OUTCOME"])
y_train = train_df["OUTCOME"].astype(int)
X_test = test_df.drop(columns=["OUTCOME"])
y_test = test_df["OUTCOME"].astype(int)

# Ensure boolean to int conversion for consistent processing
for df_set in [X_train, X_test]:
    bool_cols = df_set.select_dtypes(include=["bool"]).columns
    df_set[bool_cols] = df_set[bool_cols].astype(int)

print(f"Data Loaded: Train {X_train.shape}, Test {X_test.shape}")

# ==========================================
# 3. PHASE I: BROAD RANDOMIZED SEARCH
# ==========================================
# We use verbose=3 to get the detailed logs you requested (parameters, timing per fold)
print("\n--- PHASE I: Broad Randomized Search (Finding the Optimal Region) ---")

param_dist = {
    'n_estimators': [200, 500, 800, 1000],
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

rf_base = RandomForestClassifier(random_state=42)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# n_iter=100 ensures we cover a vast variety of combinations
random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=100,
    cv=cv_strategy,
    scoring='f1',
    verbose=3,      # Detailed logging level
    n_jobs=-1,      # Use all available CPU cores
    random_state=42
)

random_search.fit(X_train, y_train)
best_initial_params = random_search.best_params_
print(f"\n--- Phase I Best Params: {best_initial_params} ---")

# ==========================================
# 4. PHASE II: REFINED GRID SEARCH (Deep Optimization)
# ==========================================
# Tuning specifically around the best found region + adding CCP Alpha (Pruning)
print("\n--- PHASE II: Targeted Grid Search & Cost-Complexity Pruning ---")

param_grid = {
    'n_estimators': [best_initial_params['n_estimators']],
    'max_depth': [best_initial_params['max_depth']],
    'min_samples_split': [best_initial_params['min_samples_split']],
    'min_samples_leaf': [best_initial_params['min_samples_leaf']],
    'ccp_alpha': [0.0, 0.0001, 0.001, 0.005], # Fine-tuning the pruning alpha
    'class_weight': [best_initial_params['class_weight']]
}

grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1',
    verbose=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"\n--- Phase II Final Best Params: {grid_search.best_params_} ---")

# ==========================================
# 5. THRESHOLD OPTIMIZATION
# ==========================================
# Optimizing the decision threshold based on Precision-Recall Curve to maximize F1
y_proba_train = best_rf.predict_proba(X_train)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_train, y_proba_train)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimized Decision Threshold (based on Train F1): {best_threshold:.4f}")

# ==========================================
# 6. FINAL EVALUATION ON TEST SET
# ==========================================
y_proba_test = best_rf.predict_proba(X_test)[:, 1]
y_pred_opt = (y_proba_test >= best_threshold).astype(int)

p = precision_score(y_test, y_pred_opt)
r = recall_score(y_test, y_pred_opt)
f1 = f1_score(y_test, y_pred_opt)
auc = roc_auc_score(y_test, y_proba_test)

print("\n--- FINAL EVALUATION (Deep Optimized Random Forest) ---")
print(f"Precision : {p:.4f}")
print(f"Recall    : {r:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred_opt))

# ==========================================
# 7. VISUALIZATION
# ==========================================

# 7.1 Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_opt), annot=True, fmt="d", cmap="Greens")
plt.title(f"Optimized RF Confusion Matrix (Threshold: {best_threshold:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/rf_optimized_cm.png")
plt.show()

# 7.2 Feature Importance (Sorted)
feat_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 8))
feat_importances.plot(kind='barh', color='darkgreen', edgecolor='black')
plt.title("Top 15 Features - Feature Importance (Random Forest)")
plt.xlabel("Gini Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/rf_feature_importance.png")
plt.show()

# 7.3 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_test)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='forestgreen', lw=2, label=f'ROC AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--', alpha=0.5)
plt.title('Receiver Operating Characteristic (ROC) - Optimized RF')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(f"{PLOT_DIR}/rf_roc_curve.png")
plt.show()

# 7.4 Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color='darkred', lw=2, label='PR Curve')
plt.title('Precision-Recall Curve - Optimized RF')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(alpha=0.2)
plt.savefig(f"{PLOT_DIR}/rf_pr_curve.png")
plt.show()

print(f"\nAll Random Forest results and plots saved in '{PLOT_DIR}'.")