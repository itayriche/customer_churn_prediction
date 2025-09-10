"""Data preprocessing module for customer churn prediction."""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from .config import (
    DATA_PATH, RANDOM_STATE, TEST_SIZE, CATEGORICAL_FEATURES, 
    NUMERICAL_FEATURES, TARGET_COLUMN, ID_COLUMN
)


def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the customer churn dataset.
    
    Args:
        file_path (str, optional): Path to the CSV file. If None, uses default path.
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if file_path is None:
        file_path = DATA_PATH
    
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    # Make a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Convert TotalCharges and MonthlyCharges to numeric
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['MonthlyCharges'] = pd.to_numeric(df_clean['MonthlyCharges'], errors='coerce')
    
    # Check for missing values
    missing_before = df_clean.isnull().sum().sum()
    if missing_before > 0:
        print(f"Found {missing_before} missing values")
        print("Rows with missing data:")
        print(df_clean[df_clean.isnull().any(axis=1)])
    
    # Remove rows with missing values
    df_clean = df_clean.dropna()
    
    # Remove customerID column as it's not useful for prediction
    if ID_COLUMN in df_clean.columns:
        df_clean = df_clean.drop(ID_COLUMN, axis=1)
    
    # Convert SeniorCitizen from numeric to categorical
    df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})
    
    print(f"Data cleaned. Final shape: {df_clean.shape}")
    return df_clean


def get_feature_info(df: pd.DataFrame) -> None:
    """
    Display information about the features in the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("=== DATASET INFO ===")
    print(f"Dataset shape: {df.shape}")
    print("\n=== COLUMN TYPES ===")
    print(df.dtypes)
    
    print("\n=== UNIQUE VALUES PER COLUMN ===")
    for column in df.columns:
        unique_values = df[column].unique()
        if len(unique_values) <= 10:
            print(f"{column}: {unique_values}")
        else:
            print(f"{column}: {len(unique_values)} unique values")
    
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe(include='all'))


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target variable.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target variable
    """
    # Separate features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN].replace({'No': 0, 'Yes': 1})
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def create_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing pipeline for features.
    
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Define preprocessing steps
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    numerical_transformer = StandardScaler()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'  # Keep any other columns unchanged
    )
    
    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of data for testing
        random_state (int): Random state for reproducibility
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    print(f"Test target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def get_correlation_with_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.Series:
    """
    Calculate correlation of all features with the target variable.
    
    Args:
        df (pd.DataFrame): Dataset with all features
        target_col (str): Name of target column
        
    Returns:
        pd.Series: Correlation values sorted in descending order
    """
    # Encode categorical variables for correlation calculation
    encoded_df = pd.get_dummies(df, drop_first=True)
    
    # Get correlation with target (assuming target is binary and encoded as target_Yes)
    target_encoded = f"{target_col}_Yes"
    if target_encoded in encoded_df.columns:
        correlations = encoded_df.corr()[target_encoded].sort_values(ascending=False)
        # Remove the target itself from correlations
        correlations = correlations.drop(target_encoded)
    else:
        # If target is already numeric
        target_numeric = df[target_col].replace({'No': 0, 'Yes': 1})
        features_encoded = encoded_df.drop([col for col in encoded_df.columns if target_col in col], axis=1)
        correlations = features_encoded.corrwith(target_numeric).sort_values(ascending=False)
    
    return correlations


def preprocess_pipeline(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Complete preprocessing pipeline.
    
    Args:
        file_path (str, optional): Path to data file
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    print("Starting data preprocessing pipeline...")
    
    # Load and clean data
    df = load_data(file_path)
    df_clean = clean_data(df)
    
    # Display data info
    get_feature_info(df_clean)
    
    # Prepare features and target
    X, y = prepare_features_target(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Create preprocessor
    preprocessor = create_preprocessor()
    
    print("Preprocessing pipeline completed successfully!")
    
    return X_train, X_test, y_train, y_test, preprocessor