import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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
print("--- Loading Prepared Data for MLP (Neural Network) ---")
try:
    train_df = pd.read_excel("train_data_final.xlsx")
    test_df = pd.read_excel("test_data_final.xlsx")
except Exception:
    train_df = pd.read_csv("train_data_final.csv")
    test_df = pd.read_csv("test_data_final.csv")

X_train = train_df.drop(columns=["OUTCOME"])
y_train = train_df["OUTCOME"].astype(int)
X_test = test_df.drop(columns=["OUTCOME"])
y_test = test_df["OUTCOME"].astype(int)

# Ensure numeric types (Sklearn MLP works best with floats/ints)
for df_set in [X_train, X_test]:
    bool_cols = df_set.select_dtypes(include=["bool"]).columns
    df_set[bool_cols] = df_set[bool_cols].astype(int)

# ==========================================
# 3. DEEP ARCHITECTURE OPTIMIZATION (Grid Search)
# ==========================================
print("\n--- Starting Exhaustive MLP Optimization ---")
print("Searching for the best Neural Architecture...")

# hidden_layer_sizes: Each tuple represents (neurons_in_layer_1, neurons_in_layer_2, ...)
param_grid = {
    'hidden_layer_sizes': [(64, 32), (100, 50, 25), (128, 64), (50, 50, 50)],
    'activation': ['relu', 'tanh'],           # Non-linear activation functions
    'solver': ['adam', 'sgd'],                # Optimization algorithms
    'alpha': [0.0001, 0.001, 0.01],           # L2 penalty (Regularization) to prevent Overfitting
    'learning_rate': ['constant', 'adaptive'] # How the weights change during training
}

mlp = MLPClassifier(max_iter=1000, early_stopping=True, random_state=42)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search with detailed progress output
grid_search_mlp = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1',
    verbose=3,
    n_jobs=-1
)

grid_search_mlp.fit(X_train, y_train)
best_mlp = grid_search_mlp.best_estimator_

print(f"\n--- Best MLP Parameters Found: {grid_search_mlp.best_params_} ---")

# ==========================================
# 4. THRESHOLD OPTIMIZATION (Precision-Recall)
# ==========================================
y_proba_train = best_mlp.predict_proba(X_train)[:, 1]
p_curve, r_curve, thresholds = precision_recall_curve(y_train, y_proba_train)
f1_scores = 2 * (p_curve * r_curve) / (p_curve + r_curve + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimized Decision Threshold: {best_threshold:.4f}")

# ==========================================
# 5. FINAL EVALUATION ON TEST SET
# ==========================================
y_proba_test = best_mlp.predict_proba(X_test)[:, 1]
y_pred_opt = (y_proba_test >= best_threshold).astype(int)

print("\n--- FINAL EVALUATION (MLP Deep Learning) ---")
print(f"Precision : {precision_score(y_test, y_pred_opt):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_opt):.4f}")
print(f"F1-score  : {f1_score(y_test, y_pred_opt):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba_test):.4f}")

# ==========================================
# 6. VISUALIZATION
# ==========================================

# 6.1 Loss Curve (Specific to Neural Networks)
# Shows how the error decreased over time during training
plt.figure(figsize=(8, 6))
plt.plot(best_mlp.loss_curve_, color='purple')
plt.title('MLP Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss (Cross-Entropy)')
plt.grid(True, alpha=0.3)
plt.savefig(f"{PLOT_DIR}/mlp_loss_curve.png")
plt.show()

# 6.2 Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_opt), annot=True, fmt="d", cmap="Purples")
plt.title(f"MLP Neural Network Confusion Matrix\nThreshold: {best_threshold:.2f}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f"{PLOT_DIR}/mlp_cm.png")
plt.show()

# 6.3 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba_test)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'AUC = {roc_auc_score(y_test, y_proba_test):.3f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', alpha=0.5)
plt.title('ROC Curve: MLP Deep Learning')
plt.legend()
plt.savefig(f"{PLOT_DIR}/mlp_roc.png")
plt.show()

print(f"\nMLP optimization complete. Results saved in '{PLOT_DIR}'.")