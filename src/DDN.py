import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
# SimpleImputer is not used directly for Random Sample here, but included for context
# from sklearn.impute import SimpleImputer

# Set style for consistency
sns.set_style("whitegrid")

# --- 1. Load the dataset (Original Data) ---
file_path = "Car_Insurance_Claim 3.csv"
df_original = pd.read_csv(file_path)

# Drop irrelevant columns from the original DataFrame
cols_to_drop = ["ID", "POSTAL_CODE", "RACE"]
df_original = df_original.drop(columns=cols_to_drop, errors='ignore')

# Define columns that need imputation
IMPUTATION_COLS = ["CREDIT_SCORE", "ANNUAL_MILEAGE"]
# Define all continuous numeric columns potentially correlated with IMPUTATION_COLS
ALL_NUMERIC_COLS = IMPUTATION_COLS + [col for col in df_original.columns if
                                      col not in IMPUTATION_COLS and df_original[col].dtype in ['int64', 'float64']]


# --- 2. Function to Prepare Imputed DataFrames (All Methods for Comparison) ---

def prepare_imputed_dataframes(df):
    """Prepares multiple imputed DataFrames using different methods for comparison."""

    # 2.1 DataFrame: Single-Feature Impute (Original methods)
    df_single_impute = df.copy()

    # CREDIT_SCORE: Iterative Imputer (Single feature)
    iter_imp_single = IterativeImputer(random_state=42)
    if 'CREDIT_SCORE' in df_single_impute.columns:
        df_single_impute["CREDIT_SCORE"] = iter_imp_single.fit_transform(
            df_single_impute[["CREDIT_SCORE"]].values.reshape(-1, 1)
        ).flatten()

    # ANNUAL_MILEAGE: Median Imputation
    if 'ANNUAL_MILEAGE' in df_single_impute.columns:
        df_single_impute["ANNUAL_MILEAGE"] = df_single_impute["ANNUAL_MILEAGE"].fillna(
            df_single_impute["ANNUAL_MILEAGE"].median()
        )

    # 2.2 DataFrame: Iterative Imputer (Multi-Feature - Recommended Improvement)
    df_iterative_multi = df.copy()

    # Get only the columns present in the current DF
    cols_for_iter_multi = [col for col in ALL_NUMERIC_COLS if col in df.columns and df[col].isnull().any()]

    if cols_for_iter_multi:
        # Run Imputer on all relevant columns simultaneously to leverage correlations
        iter_imp_multi = IterativeImputer(random_state=42)

        imputed_data = iter_imp_multi.fit_transform(df_iterative_multi[ALL_NUMERIC_COLS])
        df_iterative_multi[ALL_NUMERIC_COLS] = imputed_data

    # 2.3 DataFrame: Random Sample Imputation (Alternative for Annual Mileage)
    df_random_sample = df.copy()

    # Function for Random Sample Imputation
    def impute_random_sample(series):
        missing_count = series.isnull().sum()
        if missing_count == 0:
            return series

        random_samples = series.dropna().sample(missing_count, replace=True, random_state=42)
        series_imputed = series.copy()
        series_imputed.loc[series_imputed.isnull()] = random_samples.values
        return series_imputed

    # Apply Random Sample only to ANNUAL_MILEAGE
    if 'ANNUAL_MILEAGE' in df_random_sample.columns:
        df_random_sample["ANNUAL_MILEAGE"] = impute_random_sample(df_random_sample["ANNUAL_MILEAGE"])

    # Keep CREDIT_SCORE as the (less optimal) Single Impute for a fair comparison of Annual Mileage methods
    if 'CREDIT_SCORE' in df_random_sample.columns:
        df_random_sample["CREDIT_SCORE"] = df_single_impute["CREDIT_SCORE"].copy()

    return {
        "Original (Existing Data)": df,
        "Single Impute (Median/Iterative)": df_single_impute,
        "Iterative (Multi-Feature)": df_iterative_multi,
        "Random Sample (Annual Mileage)": df_random_sample
    }


imputed_dfs = prepare_imputed_dataframes(df_original.copy())


# --- 3. Function to Plot Comparison (Up to 4 Methods) ---

def plot_multi_imputation_comparison(imputed_dataframes, column_name, methods_to_plot):
    """Displays side-by-side distribution plots comparing different imputation methods."""

    data_for_plot = {k: imputed_dataframes[k] for k in methods_to_plot if k in imputed_dataframes}
    num_plots = len(data_for_plot)

    if num_plots == 0:
        return

    # Dynamic subplot layout
    if num_plots > 2:
        rows, cols = 2, 2
    else:
        rows, cols = 1, num_plots

    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = axes.flatten()
    fig.suptitle(f'Distribution Comparison: {column_name}', fontsize=18)

    colors = ['blue', 'red', 'green', 'purple']

    for i, (method, df) in enumerate(data_for_plot.items()):
        ax = axes[i]

        if method == "Original (Existing Data)":
            series = df[column_name].dropna()
            title = f'{method} (n={len(series)})'
        else:
            series = df[column_name]
            title = method

        sns.histplot(series, kde=True, ax=ax, color=colors[i])
        ax.set_title(title)
        ax.set_xlabel(column_name)
        ax.set_ylabel('Density/Frequency')

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- 4. Execution of Comparison Plots ---

print("--- Comparison: CREDIT_SCORE ---")
plot_multi_imputation_comparison(
    imputed_dfs,
    "CREDIT_SCORE",
    methods_to_plot=["Original (Existing Data)", "Single Impute (Median/Iterative)", "Iterative (Multi-Feature)"]
)

print("\n--- Comparison: ANNUAL_MILEAGE ---")
plot_multi_imputation_comparison(
    imputed_dfs,
    "ANNUAL_MILEAGE",
    methods_to_plot=["Original (Existing Data)", "Single Impute (Median/Iterative)", "Iterative (Multi-Feature)",
                     "Random Sample (Annual Mileage)"]
)

# ####################################################################
# --- FINAL DECISION: Missing Value Imputation ---
# The best approach for preserving multivariate relationships is the
# Iterative Imputer (Multi-Feature). We select this DataFrame for modeling.
# ####################################################################

# Set the primary DataFrame (df) to the one with the chosen imputation method
df = imputed_dfs["Iterative (Multi-Feature)"].copy()

# Print confirmation
print("\n--- FINAL DATAFRAME SELECTED ---")
print("The 'df' DataFrame is now set to the Iterative (Multi-Feature) imputed data.")
print(f"Missing values in {IMPUTATION_COLS} after final selection:")
print(df[IMPUTATION_COLS].isnull().sum())

# --- START: FEATURE ENGINEERING - Outlier Handling and Normalization ---

# Define continuous/count columns for scaling (Normalization)
# We include all non-normalized numeric variables for uniform scaling (Mean=0, Std=1).
SCALING_COLS = ["ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"]


# ####################################################################
# PHASE 1: OUTLIER HANDLING (Capping/Clipping) on ANNUAL_MILEAGE
# The IQR method (Q3 + 1.5*IQR) is used for robust outlier detection.
# ####################################################################

mileage_series = df["ANNUAL_MILEAGE"]

# 1. Calculate IQR and bounds
Q1 = mileage_series.quantile(0.25)
Q3 = mileage_series.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 2. Print statistical report and create Boxplot (Pre-Clipping Visualization)
print("\n--- ANNUAL_MILEAGE Outlier Analysis (Boxplot Method) ---")
print(f"Q1 (25th Percentile): {Q1:.2f}")
print(f"Q3 (75th Percentile): {Q3:.2f}")
print(f"Upper Bound for Outliers: {upper_bound:.2f}")

plt.figure(figsize=(8, 6))
sns.boxplot(y=mileage_series, color='skyblue')
plt.title('Boxplot of ANNUAL_MILEAGE (Pre-Clipping)', fontsize=14)
plt.ylabel('Annual Mileage')
plt.show()

# 3. Clipping Action: Replace values outside the bounds with the bounds themselves (Trimming)
outlier_count = mileage_series[(mileage_series < lower_bound) | (mileage_series > upper_bound)].count()

if outlier_count > 0:
    # Applying the clipping directly to the main DataFrame 'df'
    df["ANNUAL_MILEAGE"] = np.clip(df["ANNUAL_MILEAGE"], lower_bound, upper_bound)
    print(f"\nACTION: {outlier_count} outliers in ANNUAL_MILEAGE were clipped to the bounds ({lower_bound:.2f} to {upper_bound:.2f}).")
else:
    print("\nACTION: No significant outliers found, no clipping performed.")

# Visualization after clipping (to confirm action)
plt.figure(figsize=(8, 6))
sns.boxplot(y=df["ANNUAL_MILEAGE"], color='lightgreen')
plt.title('Boxplot of ANNUAL_MILEAGE (After Clipping)', fontsize=14)
plt.ylabel('Annual Mileage')
plt.show()


# ####################################################################
# PHASE 2: NORMALIZATION (Z-Score Standardization)
# Applied to all scaling columns to ensure Mean=0 and Std_Dev=1.
# This equalizes feature contribution across all numeric features.
# ####################################################################

print("\n--- Applying Z-Score Normalization (Standardization) on all scaling columns ---")

scaler = StandardScaler()

# Check and scale the selected columns
for col in SCALING_COLS:
    if col in df.columns:
        # Scale only if the standard deviation is not zero
        if df[col].std() > 1e-6:
            df[col] = scaler.fit_transform(df[[col]])
            print(f"Standardized column: {col}")
        else:
            print(f"Skipping {col}: Standard deviation is near zero (constant values).")

# Verification: Check the mean (should be ~0) and standard deviation (should be ~1)
print("\n--- Verification of Standardization ---")
print(df[SCALING_COLS].describe().loc[['mean', 'std']].round(4))

# --- END: FEATURE ENGINEERING ---