#!/usr/bin/env python3
"""
Simple training script for web app - creates models without relative imports.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
import os

# Configuration
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_SAVE_PATH = "models/"
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

def create_models():
    """Create model instances."""
    return {
        'random_forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        'logistic_regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'svm': SVC(random_state=RANDOM_STATE, probability=True)
    }

def main():
    """Main training function."""
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
    
    # Create preprocessor
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ]
    )
    
    # Create and train models
    models = create_models()
    trained_models = {}
    
    # Ensure models directory exists
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Save model
        filename = f"{MODEL_SAVE_PATH}{name}_20241210_120000.joblib"
        joblib.dump(pipeline, filename)
        print(f"Saved: {filename}")
        
        trained_models[name] = pipeline
    
    print(f"\n✅ Successfully trained and saved {len(trained_models)} models!")
    print("Models available for web app:")
    for name in trained_models.keys():
        print(f"  • {name}")

if __name__ == "__main__":
    main()