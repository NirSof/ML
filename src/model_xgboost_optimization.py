import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import ClassifierMixin, BaseEstimator
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
train_df = pd.read_excel("train_data_final.xlsx")
test_df = pd.read_excel("test_data_final.xlsx")

X_train = train_df.drop(columns=["OUTCOME"])
y_train = train_df["OUTCOME"].astype(int)
X_test = test_df.drop(columns=["OUTCOME"])
y_test = test_df["OUTCOME"].astype(int)

# Fix for boolean/numeric types
for df in [X_train, X_test]:
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight: {pos_weight:.2f}")


# ==========================================
# 3. THE BULLETPROOF WRAPPER (Fixes Sklearn 1.6 Error)
# ==========================================
# We wrap XGBoost in a custom class that Sklearn 1.6 understands perfectly
class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        self.params = params
        self.model = XGBClassifier(**self.params)
        self.classes_ = [0, 1]

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        self.params.update(params)
        self.model = XGBClassifier(**self.params)
        return self


# ==========================================
# 4. INTENSIVE GRID SEARCH
# ==========================================
print("\n--- Starting Exhaustive Grid Search (Bypassing Tags Error) ---")

# Defining a very broad and deep grid as requested
param_grid = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 7, 10],
    'subsample': [0.8, 1.0],
    'gamma': [0.1, 0.5, 1],
    'scale_pos_weight': [1, pos_weight]
}

# Use the wrapper instead of the raw XGBClassifier
wrapped_xgb = XGBWrapper(eval_metric='logloss', random_state=42)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=wrapped_xgb,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1',
    verbose=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"\n--- Best Parameters Found: {grid_search.best_params_} ---")

# ==========================================
# 5. FINAL EVALUATION & VISUALIZATION
# ==========================================
y_proba_test = best_model.predict_proba(X_test)[:, 1]

# Threshold Optimization
p_curve, r_curve, thresholds = precision_recall_curve(y_train, best_model.predict_proba(X_train)[:, 1])
f1_scores = 2 * (p_curve * r_curve) / (p_curve + r_curve + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]

y_pred_opt = (y_proba_test >= best_threshold).astype(int)

print("\n--- FINAL EVALUATION (XGBoost Deep Optimized) ---")
print(f"F1-score: {f1_score(y_test, y_pred_opt):.4f} | ROC-AUC: {roc_auc_score(y_test, y_proba_test):.4f}")

# Plots (Confusion Matrix & Importance)
plt.figure(figsize=(7, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_opt), annot=True, fmt="d", cmap="Oranges")
plt.title(f"XGBoost Optimized (Threshold: {best_threshold:.2f})")
plt.savefig(f"{PLOT_DIR}/xgboost_final_cm.png")
plt.show()

# Importance (Accessing inner model)
importances = best_model.model.feature_importances_
feat_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 8))
feat_importances.plot(kind='barh', color='darkorange')
plt.title("XGBoost Top 15 Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/xgboost_final_importance.png")
plt.show()