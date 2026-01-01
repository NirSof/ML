import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.patches import Patch

# --- INITIAL SETTINGS ---
# Set general style and figure dimensions
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = [10, 6]

# --- 1. DATA LOADING ---
# Loading the primary dataset
file_path = "Car_Insurance_Claim 3.csv"
df_raw = pd.read_csv(file_path)

# --- 2. EXPLORATORY DATA ANALYSIS (EDA) ---
# Goal: Visualize data to identify patterns and correlations

# 2.1 Target Variable Distribution
plt.figure(figsize=(6, 6))
df_raw['OUTCOME'].value_counts().plot.pie(
    autopct='%1.1f%%',
    labels=['No Claim', 'Claim'],
    colors=['#66b3ff', '#ff9999'],
    startangle=90
)
plt.title('Figure 1: Target Variable Distribution (OUTCOME)', fontsize=14)
plt.ylabel('')
plt.show()

# 2.2 Claims by Age Group
age_order = ['16-25', '26-39', '40-64', '65+']
plt.figure(figsize=(8, 5))
sns.countplot(data=df_raw, x='AGE', hue='OUTCOME', order=age_order, palette='viridis')
plt.title('Figure 2: Claims by Age Group', fontsize=14)
plt.xlabel('Age Group')
plt.legend(title='Outcome', labels=['No Claim', 'Claim'])
plt.tight_layout()
plt.show()

# 2.3 Claims by Driving Experience
exp_order = ['0-9y', '10-19y', '20-29y', '30y+']
plt.figure(figsize=(8, 5))
sns.countplot(data=df_raw, x='DRIVING_EXPERIENCE', hue='OUTCOME', order=exp_order, palette='viridis')
plt.title('Figure 3: Claims by Driving Experience', fontsize=14)
plt.xlabel('Driving Experience (Years)')
plt.legend(title='Outcome', labels=['No Claim', 'Claim'])
plt.tight_layout()
plt.show()

# 2.4 Correlation Heatmap
plt.figure(figsize=(14, 12))
numeric_cols_for_heat = df_raw.select_dtypes(include=[np.number]).columns
correlation_matrix = df_raw[numeric_cols_for_heat].drop(columns=['ID', 'POSTAL_CODE'], errors='ignore').corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 9})
plt.title('Feature Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2.5 Top 8 Strongest Correlated Variables (Vertical Bar Chart)
correlations = correlation_matrix['OUTCOME'].drop('OUTCOME', errors='ignore')
top_corr = correlations.loc[correlations.abs().sort_values(ascending=False).index].head(8).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
colors_corr = ['#4c72b0' if x > 0 else '#c44e52' for x in top_corr.values]
plt.bar(top_corr.index, top_corr.values, color=colors_corr, edgecolor='black', alpha=0.9)
plt.title('Top 8 Strongest Correlated Variables with Outcome', fontsize=14)
plt.ylabel('Correlation Coefficient')
plt.axhline(0, color='black', linewidth=0.8)
legend_elements = [
    Patch(facecolor='#4c72b0', edgecolor='black', label='Positive Correlation'),
    Patch(facecolor='#c44e52', edgecolor='black', label='Negative Correlation')
]
plt.legend(handles=legend_elements, loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# --- 3. DATA CLEANING ---
# Dropping non-predictive columns
cols_to_drop = ["ID", "POSTAL_CODE", "RACE"]
df = df_raw.drop(columns=cols_to_drop, errors='ignore').copy()

# --- 4. MISSING VALUE IMPUTATION ANALYSIS ---
# Evaluating different strategies for missing data

IMPUTATION_COLS = ["CREDIT_SCORE", "ANNUAL_MILEAGE"]
ALL_NUMERIC_COLS = IMPUTATION_COLS + [col for col in df.columns if
                                      col not in IMPUTATION_COLS and df[col].dtype in ['int64', 'float64']]


def prepare_imputed_dataframes(df_input):
    """Prepares and stores different imputation versions for comparison."""
    results = {}
    results["Original (Existing Data)"] = df_input.copy()

    # Median
    df_median = df_input.copy()
    for col in IMPUTATION_COLS:
        df_median[col] = df_median[col].fillna(df_median[col].median())
    results["Median Imputation"] = df_median

    # Single-Feature Iterative
    df_single_iter = df_input.copy()
    for col in IMPUTATION_COLS:
        it_imp = IterativeImputer(random_state=42)
        df_single_iter[col] = it_imp.fit_transform(df_single_iter[[col]]).flatten()
    results["Single-Feature Iterative"] = df_single_iter

    # Multi-Feature Iterative (Recommended)
    df_multi_iter = df_input.copy()
    it_imp_multi = IterativeImputer(random_state=42)
    df_multi_iter[ALL_NUMERIC_COLS] = it_imp_multi.fit_transform(df_multi_iter[ALL_NUMERIC_COLS])
    results["Iterative (Multi-Feature)"] = df_multi_iter

    return results


imputed_dfs = prepare_imputed_dataframes(df)


# Imputation Comparison Plot (Restored to 2x2 layout)
def plot_imputation_comparison(imputed_dataframes, column_name, methods_to_plot):
    data_for_plot = {k: imputed_dataframes[k] for k in methods_to_plot if k in imputed_dataframes}
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle(f'Imputation Comparison: {column_name}', fontsize=18)
    colors = ['blue', 'red', 'green', 'purple']

    for i, (method, dframe) in enumerate(data_for_plot.items()):
        series = dframe[column_name].dropna() if method == "Original (Existing Data)" else dframe[column_name]
        sns.histplot(series, kde=True, ax=axes[i], color=colors[i])
        axes[i].set_title(method)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


print("--- Comparison: CREDIT_SCORE ---")
plot_imputation_comparison(imputed_dfs, "CREDIT_SCORE",
                           ["Original (Existing Data)", "Median Imputation", "Single-Feature Iterative",
                            "Iterative (Multi-Feature)"])

print("\n--- Comparison: ANNUAL_MILEAGE ---")
plot_imputation_comparison(imputed_dfs, "ANNUAL_MILEAGE",
                           ["Original (Existing Data)", "Median Imputation", "Single-Feature Iterative",
                            "Iterative (Multi-Feature)"])

# Finalizing the imputation choice
df = imputed_dfs["Iterative (Multi-Feature)"].copy()
print("\nFINAL DECISION: Iterative (Multi-Feature) selected for modeling.")

# --- 5. FEATURE ENGINEERING: OUTLIERS & SCALING ---

# 5.1 Outlier Handling for ANNUAL_MILEAGE
# Visualizing Pre-Clipping State (Vertical Boxplot)
plt.figure(figsize=(6, 8))
sns.boxplot(y=df["ANNUAL_MILEAGE"], color='skyblue')
plt.title('Boxplot of ANNUAL_MILEAGE (Pre-Clipping)', fontsize=14)
plt.show()

# Clipping logic
Q1, Q3 = df["ANNUAL_MILEAGE"].quantile([0.25, 0.75])
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR
print(f"ANNUAL_MILEAGE Clipping: Upper={upper_bound:.2f}, Lower={lower_bound:.2f}")

df["ANNUAL_MILEAGE"] = np.clip(df["ANNUAL_MILEAGE"], lower_bound, upper_bound)

# Visualizing After Clipping State (Vertical Boxplot)
plt.figure(figsize=(6, 8))
sns.boxplot(y=df["ANNUAL_MILEAGE"], color='lightgreen')
plt.title('Boxplot of ANNUAL_MILEAGE (After Clipping)', fontsize=14)
plt.show()

# 5.2 Standardization (StandardScaler)
SCALING_COLS = ["ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"]
scaler = StandardScaler()
df[SCALING_COLS] = scaler.fit_transform(df[SCALING_COLS])
print("Standardization applied to:", SCALING_COLS)

# --- 6. CATEGORICAL ENCODING ---
categorical_cols = df.select_dtypes(include=["object"]).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nOne-Hot Encoding complete. New dataset shape:", df_encoded.shape)

# --- 7. MULTICOLLINEARITY CHECK (VIF) ---
X_vif = df_encoded.drop(columns=["OUTCOME"], errors="ignore").copy()
# Ensure numeric types for boolean columns
for col in X_vif.select_dtypes(include=['bool']).columns:
    X_vif[col] = X_vif[col].astype(int)

vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("\n--- VIF Results (Sorted) ---")
print(vif_data.sort_values("VIF", ascending=False))

# --- 8. TRAIN / TEST SPLIT (80/20) ---
X = df_encoded.drop(columns=["OUTCOME"])
y = df_encoded["OUTCOME"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData Split complete: Train size={X_train.shape[0]}, Test size={X_test.shape[0]}")

# --- 9. DATA EXPORT ---
# Combining features and target for export
train_final = X_train.copy()
train_final["OUTCOME"] = y_train.values
test_final = X_test.copy()
test_final["OUTCOME"] = y_test.values

train_final.to_excel("train_data_final.xlsx", index=False)
test_final.to_excel("test_data_final.xlsx", index=False)
print("\nSuccess: Processed files saved successfully.")