import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# --- Loading data ---
file_path = "Car_Insurance_Claim 3.csv"
df = pd.read_csv(file_path)

# --- 1. אימפיוטציה ל-CREDIT_SCORE עם IterativeImputer ---
iter_imp = IterativeImputer(random_state=42)
df["CREDIT_SCORE"] = iter_imp.fit_transform(df[["CREDIT_SCORE"]])

# --- 2. אימפיוטציה ל-ANNUAL_MILEAGE עם חציון ---
df["ANNUAL_MILEAGE"] = df["ANNUAL_MILEAGE"].fillna(df["ANNUAL_MILEAGE"].median())

# בדיקה
print("Missing values after imputation:")
print(df[["CREDIT_SCORE", "ANNUAL_MILEAGE"]].isnull().sum())