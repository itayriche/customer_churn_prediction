"""Model training module for customer churn prediction."""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
import joblib
import os
from datetime import datetime

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("Warning: XGBoost not available. Install with: pip install xgboost")

from .config import MODELS_CONFIG, RANDOM_STATE, CV_FOLDS, SCORING_METRICS, MODEL_SAVE_PATH


class ModelTrainer:
    """Class for training and managing multiple machine learning models."""
    
    def __init__(self, preprocessor: ColumnTransformer):
        """
        Initialize the model trainer.
        
        Args:
            preprocessor (ColumnTransformer): Preprocessing pipeline
        """
        self.preprocessor = preprocessor
        self.models = {}
        self.trained_models = {}
        self.best_models = {}
        self.cross_val_scores = {}
        
    def _get_model_class(self, class_name: str):
        """Get model class by name."""
        model_classes = {
            'LogisticRegression': LogisticRegression,
            'RandomForestClassifier': RandomForestClassifier,
            'SVC': SVC,
            'XGBClassifier': XGBClassifier,
            'DecisionTreeClassifier': DecisionTreeClassifier,
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'AdaBoostClassifier': AdaBoostClassifier,
            'MLPClassifier': MLPClassifier
        }
        
        if class_name not in model_classes:
            raise ValueError(f"Unknown model class: {class_name}")
        
        model_class = model_classes[class_name]
        if model_class is None:
            raise ImportError(f"Model {class_name} is not available. Please install required dependencies.")
        
        return model_class
    
    def initialize_models(self, include_neural_network: bool = True) -> None:
        """
        Initialize all models with default parameters.
        
        Args:
            include_neural_network (bool): Whether to include MLPClassifier
        """
        self.models = {}
        
        for model_name, config in MODELS_CONFIG.items():
            try:
                model_class = self._get_model_class(config['class_name'])
                self.models[model_name] = model_class(**config['params'])
                print(f"Initialized {model_name}")
            except (ImportError, ValueError) as e:
                print(f"Skipping {model_name}: {e}")
        
        # Add neural network if requested
        if include_neural_network:
            try:
                self.models['neural_network'] = MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
                print("Initialized neural_network")
            except Exception as e:
                print(f"Could not initialize neural network: {e}")
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Pipeline]:
        """
        Train all initialized models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            Dict[str, Pipeline]: Dictionary of trained model pipelines
        """
        print("Training models...")
        self.trained_models = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            try:
                # Create pipeline with preprocessor and model
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('classifier', model)
                ])
                
                # Train the model
                pipeline.fit(X_train, y_train)
                self.trained_models[name] = pipeline
                
                print(f"✓ {name} trained successfully")
                
            except Exception as e:
                print(f"✗ Error training {name}: {e}")
        
        return self.trained_models
    
    def perform_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series,
                               scoring: str = 'accuracy', cv: int = CV_FOLDS) -> Dict[str, float]:
        """
        Perform cross-validation on all models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            scoring (str): Scoring metric
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, float]: Cross-validation scores for each model
        """
        print(f"Performing {cv}-fold cross-validation with {scoring} scoring...")
        cv_scores = {}
        
        for name, model in self.models.items():
            try:
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('classifier', model)
                ])
                
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring)
                cv_scores[name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
                
                print(f"{name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
                
            except Exception as e:
                print(f"Error in cross-validation for {name}: {e}")
                cv_scores[name] = {'mean': 0, 'std': 0, 'scores': []}
        
        # Store for later use
        if scoring not in self.cross_val_scores:
            self.cross_val_scores[scoring] = {}
        self.cross_val_scores[scoring] = cv_scores
        
        return cv_scores
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            model_names: Optional[List[str]] = None,
                            scoring: str = 'roc_auc', cv: int = 3) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_names (List[str], optional): Models to tune. If None, tunes all models.
            scoring (str): Scoring metric for optimization
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Best parameters and scores for each model
        """
        print(f"Performing hyperparameter tuning with {scoring} scoring...")
        
        if model_names is None:
            model_names = list(self.models.keys())
        
        tuning_results = {}
        self.best_models = {}
        
        for name in model_names:
            if name not in self.models:
                print(f"Model {name} not found. Skipping...")
                continue
                
            if name not in MODELS_CONFIG:
                print(f"No hyperparameter config for {name}. Skipping...")
                continue
            
            print(f"Tuning {name}...")
            
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('classifier', self.models[name])
                ])
                
                # Prepare parameter grid for pipeline
                param_grid = {}
                for param, values in MODELS_CONFIG[name]['hyperparameters'].items():
                    param_grid[f'classifier__{param}'] = values
                
                # Perform grid search
                grid_search = GridSearchCV(
                    pipeline, 
                    param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Store results
                tuning_results[name] = {
                    'best_score': grid_search.best_score_,
                    'best_params': grid_search.best_params_,
                    'cv_results': grid_search.cv_results_
                }
                
                self.best_models[name] = grid_search.best_estimator_
                
                print(f"✓ {name} - Best {scoring}: {grid_search.best_score_:.4f}")
                print(f"  Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"✗ Error tuning {name}: {e}")
                tuning_results[name] = {'error': str(e)}
        
        return tuning_results
    
    def get_feature_importance(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Get feature importance for a trained model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: Feature importance dataframe or None
        """
        if model_name not in self.trained_models:
            print(f"Model {model_name} not found in trained models.")
            return None
        
        model = self.trained_models[model_name]
        classifier = model.named_steps['classifier']
        
        # Check if model has feature importance
        if hasattr(classifier, 'feature_importances_'):
            # Get feature names after preprocessing
            preprocessor = model.named_steps['preprocessor']
            
            try:
                # Get feature names from the preprocessor
                feature_names = []
                
                # Numerical features
                feature_names.extend(preprocessor.named_transformers_['num'].get_feature_names_out())
                
                # Categorical features
                if 'cat' in preprocessor.named_transformers_:
                    feature_names.extend(preprocessor.named_transformers_['cat'].get_feature_names_out())
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': classifier.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df
                
            except Exception as e:
                print(f"Error extracting feature names: {e}")
                return pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(classifier.feature_importances_))],
                    'importance': classifier.feature_importances_
                }).sort_values('importance', ascending=False)
        
        elif hasattr(classifier, 'coef_'):
            # For linear models
            try:
                preprocessor = model.named_steps['preprocessor']
                feature_names = []
                
                # Get feature names
                feature_names.extend(preprocessor.named_transformers_['num'].get_feature_names_out())
                if 'cat' in preprocessor.named_transformers_:
                    feature_names.extend(preprocessor.named_transformers_['cat'].get_feature_names_out())
                
                # Use absolute values of coefficients
                coef_abs = np.abs(classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_)
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coef_abs
                }).sort_values('importance', ascending=False)
                
                return importance_df
                
            except Exception as e:
                print(f"Error extracting coefficients: {e}")
                return None
        else:
            print(f"Model {model_name} does not have feature importance or coefficients.")
            return None
    
    def save_models(self, model_names: Optional[List[str]] = None, 
                   save_path: str = MODEL_SAVE_PATH) -> Dict[str, str]:
        """
        Save trained models to disk.
        
        Args:
            model_names (List[str], optional): Names of models to save. If None, saves all.
            save_path (str): Directory to save models
            
        Returns:
            Dict[str, str]: Dictionary mapping model names to saved file paths
        """
        if model_names is None:
            models_to_save = self.trained_models
        else:
            models_to_save = {name: model for name, model in self.trained_models.items() 
                            if name in model_names}
        
        saved_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in models_to_save.items():
            try:
                filename = f"{name}_{timestamp}.joblib"
                filepath = os.path.join(save_path, filename)
                joblib.dump(model, filepath)
                saved_paths[name] = filepath
                print(f"Saved {name} to {filepath}")
            except Exception as e:
                print(f"Error saving {name}: {e}")
        
        return saved_paths
    
    def load_model(self, filepath: str) -> Pipeline:
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Pipeline: Loaded model pipeline
        """
        try:
            model = joblib.load(filepath)
            print(f"Loaded model from {filepath}")
            return model
        except Exception as e:
            raise Exception(f"Error loading model from {filepath}: {e}")


def train_baseline_models(X_train: pd.DataFrame, y_train: pd.Series, 
                         preprocessor: ColumnTransformer) -> Tuple[ModelTrainer, Dict[str, float]]:
    """
    Train baseline models without hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        preprocessor (ColumnTransformer): Preprocessing pipeline
        
    Returns:
        Tuple[ModelTrainer, Dict[str, float]]: Trained model trainer and cross-validation scores
    """
    print("=== TRAINING BASELINE MODELS ===")
    
    # Initialize trainer
    trainer = ModelTrainer(preprocessor)
    trainer.initialize_models()
    
    # Train models
    trained_models = trainer.train_models(X_train, y_train)
    
    # Perform cross-validation
    cv_scores = trainer.perform_cross_validation(X_train, y_train, scoring='roc_auc')
    
    return trainer, cv_scores


def main():
    """Main function for standalone execution."""
    from .data_preprocessing import preprocess_pipeline
    
    print("Starting model training pipeline...")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline()
    
    # Train baseline models
    trainer, cv_scores = train_baseline_models(X_train, y_train, preprocessor)
    
    # Save models
    saved_paths = trainer.save_models()
    
    print("Model training completed successfully!")
    return trainer, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()