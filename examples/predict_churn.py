#!/usr/bin/env python3
"""
Customer churn prediction script for making predictions on new data.

This script demonstrates how to:
1. Load a trained model
2. Preprocess new data using the same pipeline
3. Make predictions
4. Interpret results with confidence scores

Usage:
    python examples/predict_churn.py --model-path models/random_forest_20241210_120000.joblib
    python examples/predict_churn.py --data new_customers.csv
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import joblib

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import clean_data, create_preprocessor
from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN
from src.utils import print_section_header


class ChurnPredictor:
    """Class for making churn predictions on new data."""
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the trained model from disk."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"✓ Model loaded successfully from: {self.model_path}")
            
            # Verify model structure
            if hasattr(self.model, 'named_steps'):
                steps = list(self.model.named_steps.keys())
                print(f"  Pipeline steps: {steps}")
            else:
                print("  Model type: Single estimator (not a pipeline)")
                
        except Exception as e:
            raise Exception(f"Error loading model from {self.model_path}: {e}")
    
    def preprocess_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess new data using the same pipeline as training data.
        
        Args:
            df (pd.DataFrame): New customer data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        print("Preprocessing new data...")
        
        # Make a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Remove target column if present (for cases where we have labels)
        if TARGET_COLUMN in df_processed.columns:
            print(f"  Removing target column: {TARGET_COLUMN}")
            df_processed = df_processed.drop(TARGET_COLUMN, axis=1)
        
        # Apply the same cleaning steps as training data
        df_processed = clean_data(df_processed)
        
        # Ensure all required features are present
        missing_features = []
        for feature in CATEGORICAL_FEATURES + NUMERICAL_FEATURES:
            if feature not in df_processed.columns:
                missing_features.append(feature)
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Reorder columns to match training data
        expected_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        df_processed = df_processed[expected_columns]
        
        print(f"✓ Data preprocessed. Shape: {df_processed.shape}")
        return df_processed
    
    def predict(self, df: pd.DataFrame, return_probabilities: bool = True) -> Dict[str, Any]:
        """
        Make churn predictions on new data.
        
        Args:
            df (pd.DataFrame): New customer data
            return_probabilities (bool): Whether to return prediction probabilities
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess data
        df_processed = self.preprocess_new_data(df)
        
        # Make predictions
        print("Making predictions...")
        predictions = self.model.predict(df_processed)
        
        results = {
            'predictions': predictions,
            'prediction_labels': ['No Churn' if p == 0 else 'Churn' for p in predictions]
        }
        
        # Get prediction probabilities if available
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(df_processed)
            results['probabilities'] = probabilities
            results['churn_probability'] = probabilities[:, 1]
            results['confidence'] = np.max(probabilities, axis=1)
        
        print(f"✓ Predictions completed for {len(df_processed)} customers")
        
        return results
    
    def predict_single_customer(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single customer.
        
        Args:
            customer_data (Dict[str, Any]): Customer feature values
            
        Returns:
            Dict[str, Any]: Prediction result
        """
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Make prediction
        results = self.predict(df, return_probabilities=True)
        
        # Extract single customer result
        single_result = {
            'prediction': results['predictions'][0],
            'prediction_label': results['prediction_labels'][0]
        }
        
        if 'churn_probability' in results:
            single_result['churn_probability'] = results['churn_probability'][0]
            single_result['confidence'] = results['confidence'][0]
        
        return single_result
    
    def create_prediction_report(self, df: pd.DataFrame, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create a detailed prediction report.
        
        Args:
            df (pd.DataFrame): Customer data
            output_path (str, optional): Path to save the report
            
        Returns:
            pd.DataFrame: Prediction report
        """
        # Make predictions
        results = self.predict(df, return_probabilities=True)
        
        # Create report DataFrame
        report_df = df.copy()
        
        # Add predictions
        report_df['Predicted_Churn'] = results['prediction_labels']
        
        if 'churn_probability' in results:
            report_df['Churn_Probability'] = results['churn_probability']
            report_df['Confidence'] = results['confidence']
            
            # Add risk categories
            report_df['Risk_Category'] = pd.cut(
                results['churn_probability'],
                bins=[0, 0.3, 0.7, 1.0],
                labels=['Low Risk', 'Medium Risk', 'High Risk'],
                include_lowest=True
            )
        
        # Sort by churn probability (highest first)
        if 'Churn_Probability' in report_df.columns:
            report_df = report_df.sort_values('Churn_Probability', ascending=False)
        
        # Save report if path provided
        if output_path:
            report_df.to_csv(output_path, index=False)
            print(f"✓ Prediction report saved to: {output_path}")
        
        return report_df


def create_sample_customer_data() -> pd.DataFrame:
    """Create sample customer data for demonstration."""
    sample_data = {
        'gender': ['Female', 'Male', 'Female', 'Male'],
        'SeniorCitizen': ['No', 'Yes', 'No', 'No'],
        'Partner': ['Yes', 'No', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes', 'No'],
        'tenure': [12, 45, 2, 24],
        'PhoneService': ['Yes', 'Yes', 'Yes', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'Yes', 'No'],
        'OnlineBackup': ['No', 'Yes', 'Yes', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No', 'Yes'],
        'TechSupport': ['Yes', 'No', 'No', 'Yes'],
        'StreamingTV': ['No', 'Yes', 'No', 'Yes'],
        'StreamingMovies': ['No', 'Yes', 'No', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'Two year'],
        'PaperlessBilling': ['Yes', 'Yes', 'No', 'No'],
        'PaymentMethod': ['Electronic check', 'Credit card (automatic)', 'Mailed check', 'Bank transfer (automatic)'],
        'MonthlyCharges': [29.85, 89.10, 53.85, 74.90],
        'TotalCharges': [358.20, 4008.95, 107.70, 1798.60]
    }
    
    return pd.DataFrame(sample_data)


def demonstrate_predictions(predictor: ChurnPredictor) -> None:
    """Demonstrate prediction functionality with sample data."""
    print_section_header("PREDICTION DEMONSTRATION")
    
    # Create sample data
    sample_df = create_sample_customer_data()
    print("Sample customer data:")
    print(sample_df)
    print()
    
    # Make predictions
    report_df = predictor.create_prediction_report(sample_df)
    
    print("Prediction Results:")
    print("=" * 80)
    
    # Display key columns
    display_columns = ['gender', 'tenure', 'Contract', 'MonthlyCharges', 
                      'Predicted_Churn', 'Churn_Probability', 'Risk_Category']
    available_columns = [col for col in display_columns if col in report_df.columns]
    
    print(report_df[available_columns].to_string(index=False))
    print()
    
    # Summary statistics
    if 'Churn_Probability' in report_df.columns:
        churn_count = (report_df['Predicted_Churn'] == 'Churn').sum()
        avg_churn_prob = report_df['Churn_Probability'].mean()
        
        print(f"Summary:")
        print(f"• Total customers: {len(report_df)}")
        print(f"• Predicted to churn: {churn_count}")
        print(f"• Average churn probability: {avg_churn_prob:.3f}")
        print(f"• Risk distribution:")
        if 'Risk_Category' in report_df.columns:
            risk_dist = report_df['Risk_Category'].value_counts()
            for risk, count in risk_dist.items():
                print(f"  - {risk}: {count}")


def main():
    """Main function for script execution."""
    parser = argparse.ArgumentParser(description='Make churn predictions on new customer data')
    parser.add_argument('--model-path', type=str, 
                       help='Path to trained model file')
    parser.add_argument('--data', type=str,
                       help='Path to CSV file with customer data')
    parser.add_argument('--output', type=str,
                       help='Path to save prediction report')
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with sample data')
    
    args = parser.parse_args()
    
    # Find model path if not provided
    if not args.model_path:
        models_dir = "models"
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') and 'metadata' not in f]
            if model_files:
                # Use the most recent model
                model_files.sort(reverse=True)
                args.model_path = os.path.join(models_dir, model_files[0])
                print(f"Using model: {args.model_path}")
            else:
                print("No trained models found. Please train a model first using examples/train_model.py")
                return
        else:
            print("Models directory not found. Please train a model first using examples/train_model.py")
            return
    
    try:
        # Initialize predictor
        print_section_header("LOADING TRAINED MODEL")
        predictor = ChurnPredictor(args.model_path)
        
        if args.demo or not args.data:
            # Run demonstration
            demonstrate_predictions(predictor)
        
        if args.data:
            # Load and predict on provided data
            print_section_header("PREDICTING ON PROVIDED DATA")
            
            if not os.path.exists(args.data):
                print(f"❌ Data file not found: {args.data}")
                return
            
            print(f"Loading data from: {args.data}")
            df = pd.read_csv(args.data)
            print(f"Loaded {len(df)} customers")
            
            # Create prediction report
            output_path = args.output or f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            report_df = predictor.create_prediction_report(df, output_path)
            
            # Display summary
            print("\nPrediction Summary:")
            print("=" * 50)
            
            if 'Predicted_Churn' in report_df.columns:
                churn_summary = report_df['Predicted_Churn'].value_counts()
                print("Churn Predictions:")
                for label, count in churn_summary.items():
                    percentage = (count / len(report_df)) * 100
                    print(f"  {label}: {count} ({percentage:.1f}%)")
            
            if 'Risk_Category' in report_df.columns:
                print("\nRisk Distribution:")
                risk_summary = report_df['Risk_Category'].value_counts()
                for risk, count in risk_summary.items():
                    percentage = (count / len(report_df)) * 100
                    print(f"  {risk}: {count} ({percentage:.1f}%)")
        
        print("\n🎉 Prediction completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in prediction: {e}")
        raise


if __name__ == "__main__":
    main()