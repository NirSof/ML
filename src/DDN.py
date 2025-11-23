import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# --- 1. Load the dataset ---
file_path = "Car_Insurance_Claim 3.csv"
df = pd.read_csv(file_path)

# --- 2. Drop irrelevant or non-predictive columns ---
# ID is a unique identifier (not useful for modeling)
# POSTAL_CODE has too many outliers and no real predictive value
df = df.drop(columns=["ID", "POSTAL_CODE"])

# --- 3. Impute missing values for CREDIT_SCORE using Iterative Imputer ---
# This method predicts missing values based on other correlated features
iter_imp = IterativeImputer(random_state=42)
df["CREDIT_SCORE"] = iter_imp.fit_transform(df[["CREDIT_SCORE"]])

# --- 4. Impute missing values for ANNUAL_MILEAGE using the median ---
# Median is robust and works well here because the distribution is nearly normal
df["ANNUAL_MILEAGE"] = df["ANNUAL_MILEAGE"].fillna(df["ANNUAL_MILEAGE"].median())

# --- 5. Check missing values after imputation ---
print("Missing values after imputation:")
print(df[["CREDIT_SCORE", "ANNUAL_MILEAGE"]].isnull().sum())
