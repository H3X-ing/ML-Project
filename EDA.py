# eda_cleaning.py

import pandas as pd
import numpy as np

# Load datasets
insurance = pd.read_csv(r'C:\Users\sanie\Documents\Nastp\medicalcost\insurance.csv')
titanic = pd.read_csv(r'C:\Users\sanie\Documents\Nastp\titanic\train.csv')

# ----------- EDA Function ------------
def analyze_dataset(df, name):
    print(f"\nDATASET: {name}")
    print("-" * 40)
    print(f"Total Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nOutliers (Z-score > 3):")
    numeric_cols = df.select_dtypes(include=[np.number])
    z_scores = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
    outliers = (z_scores.abs() > 3).sum()
    print(outliers)

# ---------- Before Cleaning ----------
analyze_dataset(titanic, "Titanic (Before Cleaning)")
analyze_dataset(insurance, "Insurance (Before Cleaning)")

# ---------- Cleaning ----------
titanic_cleaned = titanic.copy()
titanic_cleaned['Age'] = titanic_cleaned['Age'].fillna(titanic_cleaned['Age'].median())
titanic_cleaned['Embarked'] = titanic_cleaned['Embarked'].fillna(titanic_cleaned['Embarked'].mode()[0])

insurance_cleaned = insurance.drop_duplicates()

# ---------- After Cleaning ----------
analyze_dataset(titanic_cleaned, "Titanic (After Cleaning)")
analyze_dataset(insurance_cleaned, "Insurance (After Cleaning)")
