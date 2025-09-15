#!/usr/bin/env python3
"""
Create data splits for webapp testing
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

# Configuration
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

def clean_data(df):
    """Clean the dataset."""
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing TotalCharges
    df = df.dropna(subset=['TotalCharges'])
    
    # Convert SeniorCitizen to string
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Drop customerID column if present
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    return df

def main():
    """Create and save data splits."""
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    df_cleaned = clean_data(df)
    
    # Prepare features and target
    X = df_cleaned.drop('Churn', axis=1)
    y = df_cleaned['Churn'].map({'No': 0, 'Yes': 1})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Save splits
    os.makedirs("data", exist_ok=True)
    splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    
    joblib.dump(splits, "data/splits.joblib")
    print("✅ Data splits saved to data/splits.joblib")
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

if __name__ == "__main__":
    main()