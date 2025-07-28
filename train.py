# model_predict.py

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

# Load cleaned datasets
insurance = pd.read_csv(r'C:\Users\sanie\Documents\Nastp\medicalcost\insurance.csv').drop_duplicates()
titanic = pd.read_csv(r'C:\Users\sanie\Documents\Nastp\titanic\train.csv')
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])

# ---------------- Titanic Classification ------------------
X_titanic = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y_titanic = titanic['Survived']

numeric_features_titanic = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features_titanic = ['Sex', 'Pclass', 'Embarked']

numeric_transformer_titanic = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer_titanic = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor_titanic = ColumnTransformer([
    ('num', numeric_transformer_titanic, numeric_features_titanic),
    ('cat', categorical_transformer_titanic, categorical_features_titanic)
])

model_titanic = Pipeline([
    ('preprocessor', preprocessor_titanic),
    ('classifier', RandomForestClassifier(random_state=42))
])

scores_titanic = cross_val_score(model_titanic, X_titanic, y_titanic, cv=5, scoring='accuracy')
print("\nTitanic Model Evaluation")
print("Accuracy Scores:", scores_titanic)
print("Average Accuracy:", scores_titanic.mean())

model_titanic.fit(X_titanic, y_titanic)

# ---------------- Insurance Regression -------------------
X_insurance = insurance.drop('charges', axis=1)
y_insurance = insurance['charges']

categorical_features_insurance = ['sex', 'smoker', 'region']
numeric_features_insurance = ['age', 'bmi', 'children']

categorical_transformer_insurance = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_transformer_insurance = Pipeline([
    ('scaler', StandardScaler())
])

preprocessor_insurance = ColumnTransformer([
    ('cat', categorical_transformer_insurance, categorical_features_insurance),
    ('num', numeric_transformer_insurance, numeric_features_insurance)
])

regression_model = Pipeline([
    ('preprocessing', preprocessor_insurance),
    ('regressor', LinearRegression())
])

mse_scores = -cross_val_score(regression_model, X_insurance, y_insurance, cv=5, scoring='neg_mean_squared_error')
print("\nInsurance Model Evaluation")
print("MSE Scores:", mse_scores)
print("Average MSE:", mse_scores.mean())

regression_model.fit(X_insurance, y_insurance)

# ---------------------- INTERACTIVE INPUT ------------------------
def predict_titanic_sample():
    print("\n--- Predict Titanic Survival ---")
    sample = {
        'Pclass': int(input("Pclass (1, 2, 3): ")),
        'Sex': input("Sex (male/female): "),
        'Age': float(input("Age: ")),
        'SibSp': int(input("Siblings/Spouses aboard: ")),
        'Parch': int(input("Parents/Children aboard: ")),
        'Fare': float(input("Fare: ")),
        'Embarked': input("Embarked (C/Q/S): ")
    }
    sample_df = pd.DataFrame([sample])
    prediction = model_titanic.predict(sample_df)
    print("Prediction:", "Survived" if prediction[0] == 1 else "Did not survive")

def predict_insurance_sample():
    print("\n--- Predict Insurance Charges ---")
    sample = {
        'age': int(input("Age: ")),
        'sex': input("Sex (male/female): "),
        'bmi': float(input("BMI: ")),
        'children': int(input("Number of children: ")),
        'smoker': input("Smoker (yes/no): "),
        'region': input("Region (northeast/northwest/southeast/southwest): ")
    }
    sample_df = pd.DataFrame([sample])
    prediction = regression_model.predict(sample_df)
    print(f"Predicted Insurance Charges: ${prediction[0]:.2f}")

# ---------------------- MENU ------------------------
def menu():
    while True:
        print("\nChoose an option:")
        print("1. Predict Titanic Survival")
        print("2. Predict Insurance Charges")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            predict_titanic_sample()
        elif choice == '2':
            predict_insurance_sample()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid input. Try again.")

menu()
