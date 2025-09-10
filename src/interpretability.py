"""Model interpretability module using SHAP for customer churn prediction."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    shap.initjs()
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

from sklearn.pipeline import Pipeline
from .config import FIGURE_SIZE


class ModelInterpreter:
    """Class for interpreting machine learning models using SHAP."""
    
    def __init__(self):
        """Initialize the interpreter."""
        self.explainers = {}
        self.shap_values = {}
        self.feature_names = []
    
    def create_explainer(self, model: Pipeline, X_train: pd.DataFrame, 
                        model_name: str, explainer_type: str = 'auto') -> None:
        """
        Create SHAP explainer for a model.
        
        Args:
            model (Pipeline): Trained model pipeline
            X_train (pd.DataFrame): Training data for background
            model_name (str): Name of the model
            explainer_type (str): Type of SHAP explainer ('auto', 'tree', 'linear', 'kernel')
        """
        if not SHAP_AVAILABLE:
            print("SHAP is not available. Please install it to use interpretability features.")
            return
        
        try:
            print(f"Creating SHAP explainer for {model_name}...")
            
            # Transform training data using the preprocessor
            X_transformed = model.named_steps['preprocessor'].transform(X_train)
            
            # Get feature names after preprocessing
            self.feature_names = self._get_feature_names(model)
            
            # Get the classifier
            classifier = model.named_steps['classifier']
            
            # Determine explainer type automatically if not specified
            if explainer_type == 'auto':
                explainer_type = self._determine_explainer_type(classifier)
            
            # Create appropriate explainer
            if explainer_type == 'tree':
                explainer = shap.TreeExplainer(classifier)
            elif explainer_type == 'linear':
                explainer = shap.LinearExplainer(classifier, X_transformed)
            elif explainer_type == 'kernel':
                # Use a sample for kernel explainer to speed up computation
                background_sample = shap.sample(X_transformed, min(100, len(X_transformed)))
                explainer = shap.KernelExplainer(classifier.predict_proba, background_sample)
            else:
                # Try tree explainer first, fallback to kernel
                try:
                    explainer = shap.TreeExplainer(classifier)
                except Exception:
                    background_sample = shap.sample(X_transformed, min(100, len(X_transformed)))
                    explainer = shap.KernelExplainer(classifier.predict_proba, background_sample)
            
            self.explainers[model_name] = explainer
            print(f"✓ SHAP explainer created for {model_name} (type: {explainer_type})")
            
        except Exception as e:
            print(f"✗ Error creating SHAP explainer for {model_name}: {e}")
    
    def _determine_explainer_type(self, classifier) -> str:
        """Determine the appropriate SHAP explainer type based on the model."""
        model_type = type(classifier).__name__
        
        if 'Tree' in model_type or 'Forest' in model_type or 'XGB' in model_type or 'Gradient' in model_type:
            return 'tree'
        elif 'Linear' in model_type or 'Logistic' in model_type:
            return 'linear'
        else:
            return 'kernel'
    
    def _get_feature_names(self, model: Pipeline) -> List[str]:
        """Get feature names after preprocessing."""
        try:
            preprocessor = model.named_steps['preprocessor']
            feature_names = []
            
            # Get numerical feature names
            if 'num' in preprocessor.named_transformers_:
                num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
                feature_names.extend(num_features)
            
            # Get categorical feature names
            if 'cat' in preprocessor.named_transformers_:
                cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
                feature_names.extend(cat_features)
            
            return feature_names
            
        except Exception as e:
            print(f"Warning: Could not extract feature names: {e}")
            return [f"feature_{i}" for i in range(model.named_steps['classifier'].n_features_in_)]
    
    def calculate_shap_values(self, model: Pipeline, X_test: pd.DataFrame, 
                            model_name: str, max_samples: int = 1000) -> Optional[np.ndarray]:
        """
        Calculate SHAP values for test data.
        
        Args:
            model (Pipeline): Trained model pipeline
            X_test (pd.DataFrame): Test data
            model_name (str): Name of the model
            max_samples (int): Maximum number of samples to explain
            
        Returns:
            np.ndarray: SHAP values or None if error
        """
        if not SHAP_AVAILABLE or model_name not in self.explainers:
            return None
        
        try:
            print(f"Calculating SHAP values for {model_name}...")
            
            # Limit samples for computational efficiency
            if len(X_test) > max_samples:
                X_sample = X_test.sample(n=max_samples, random_state=42)
                print(f"  Using sample of {max_samples} instances")
            else:
                X_sample = X_test
            
            # Transform test data
            X_transformed = model.named_steps['preprocessor'].transform(X_sample)
            
            # Calculate SHAP values
            explainer = self.explainers[model_name]
            
            if hasattr(explainer, 'shap_values'):
                shap_values = explainer.shap_values(X_transformed)
                # For binary classification, use positive class SHAP values
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values = shap_values[1]
            else:
                shap_values = explainer(X_transformed)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values
                    # For binary classification, use positive class SHAP values
                    if shap_values.ndim == 3:
                        shap_values = shap_values[:, :, 1]
            
            self.shap_values[model_name] = {
                'values': shap_values,
                'data': X_transformed,
                'feature_names': self.feature_names
            }
            
            print(f"✓ SHAP values calculated for {model_name}")
            return shap_values
            
        except Exception as e:
            print(f"✗ Error calculating SHAP values for {model_name}: {e}")
            return None
    
    def plot_summary(self, model_name: str, plot_type: str = 'bar', 
                    max_features: int = 20, figsize: Tuple[int, int] = FIGURE_SIZE) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            model_name (str): Name of the model
            plot_type (str): Type of plot ('bar', 'dot', 'violin')
            max_features (int): Maximum number of features to show
            figsize (Tuple[int, int]): Figure size
        """
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        try:
            plt.figure(figsize=figsize)
            
            shap_data = self.shap_values[model_name]
            
            if plot_type == 'bar':
                shap.summary_plot(
                    shap_data['values'], 
                    features=shap_data['data'],
                    feature_names=shap_data['feature_names'],
                    plot_type='bar',
                    max_display=max_features,
                    show=False
                )
            elif plot_type == 'dot':
                shap.summary_plot(
                    shap_data['values'], 
                    features=shap_data['data'],
                    feature_names=shap_data['feature_names'],
                    max_display=max_features,
                    show=False
                )
            elif plot_type == 'violin':
                shap.summary_plot(
                    shap_data['values'], 
                    features=shap_data['data'],
                    feature_names=shap_data['feature_names'],
                    plot_type='violin',
                    max_display=max_features,
                    show=False
                )
            
            plt.title(f'SHAP Summary Plot - {model_name}')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating SHAP summary plot for {model_name}: {e}")
    
    def plot_waterfall(self, model_name: str, instance_idx: int = 0,
                      figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Create SHAP waterfall plot for a single instance.
        
        Args:
            model_name (str): Name of the model
            instance_idx (int): Index of instance to explain
            figsize (Tuple[int, int]): Figure size
        """
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        try:
            plt.figure(figsize=figsize)
            
            shap_data = self.shap_values[model_name]
            
            # Create explanation object
            shap_values = shap_data['values'][instance_idx]
            base_value = np.mean(shap_data['values'])
            
            # Create waterfall plot
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values,
                    base_values=base_value,
                    data=shap_data['data'][instance_idx],
                    feature_names=shap_data['feature_names']
                ),
                show=False
            )
            
            plt.title(f'SHAP Waterfall Plot - {model_name} (Instance {instance_idx})')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating SHAP waterfall plot: {e}")
    
    def plot_dependence(self, model_name: str, feature_name: str,
                       interaction_feature: Optional[str] = None,
                       figsize: Tuple[int, int] = FIGURE_SIZE) -> None:
        """
        Create SHAP dependence plot for a feature.
        
        Args:
            model_name (str): Name of the model
            feature_name (str): Name of feature to analyze
            interaction_feature (str, optional): Feature to use for interaction coloring
            figsize (Tuple[int, int]): Figure size
        """
        if not SHAP_AVAILABLE or model_name not in self.shap_values:
            print(f"SHAP values not available for {model_name}")
            return
        
        try:
            plt.figure(figsize=figsize)
            
            shap_data = self.shap_values[model_name]
            
            # Find feature index
            if feature_name in shap_data['feature_names']:
                feature_idx = shap_data['feature_names'].index(feature_name)
            else:
                print(f"Feature '{feature_name}' not found in feature names")
                return
            
            # Create dependence plot
            shap.dependence_plot(
                feature_idx,
                shap_data['values'],
                features=shap_data['data'],
                feature_names=shap_data['feature_names'],
                interaction_index=interaction_feature,
                show=False
            )
            
            plt.title(f'SHAP Dependence Plot - {model_name} - {feature_name}')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error creating SHAP dependence plot: {e}")
    
    def get_feature_importance_ranking(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Get feature importance ranking based on mean absolute SHAP values.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            pd.DataFrame: Feature importance ranking
        """
        if model_name not in self.shap_values:
            return None
        
        try:
            shap_data = self.shap_values[model_name]
            
            # Calculate mean absolute SHAP values
            mean_shap_values = np.abs(shap_data['values']).mean(axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': shap_data['feature_names'],
                'importance': mean_shap_values
            }).sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error calculating SHAP feature importance: {e}")
            return None
    
    def explain_prediction(self, model: Pipeline, customer_data: pd.DataFrame,
                          model_name: str, top_features: int = 10) -> Dict[str, Any]:
        """
        Explain a single prediction with SHAP values.
        
        Args:
            model (Pipeline): Trained model
            customer_data (pd.DataFrame): Single customer data (1 row)
            model_name (str): Name of the model
            top_features (int): Number of top contributing features to show
            
        Returns:
            Dict[str, Any]: Explanation results
        """
        if not SHAP_AVAILABLE or model_name not in self.explainers:
            return {}
        
        try:
            # Transform data
            X_transformed = model.named_steps['preprocessor'].transform(customer_data)
            
            # Get prediction
            prediction = model.predict(customer_data)[0]
            prediction_proba = model.predict_proba(customer_data)[0]
            
            # Calculate SHAP values for this instance
            explainer = self.explainers[model_name]
            
            if hasattr(explainer, 'shap_values'):
                shap_vals = explainer.shap_values(X_transformed)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # Positive class
            else:
                shap_vals = explainer(X_transformed)
                if hasattr(shap_vals, 'values'):
                    shap_vals = shap_vals.values
                    if shap_vals.ndim == 3:
                        shap_vals = shap_vals[:, :, 1]  # Positive class
            
            # Get feature contributions
            contributions = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': shap_vals[0],
                'feature_value': X_transformed[0]
            })
            
            # Sort by absolute SHAP value
            contributions['abs_shap'] = np.abs(contributions['shap_value'])
            contributions = contributions.sort_values('abs_shap', ascending=False)
            
            # Get top contributing features
            top_contributions = contributions.head(top_features)
            
            explanation = {
                'prediction': int(prediction),
                'prediction_label': 'Churn' if prediction == 1 else 'No Churn',
                'churn_probability': prediction_proba[1],
                'confidence': np.max(prediction_proba),
                'top_contributions': top_contributions[['feature', 'shap_value', 'feature_value']].to_dict('records'),
                'explanation_text': self._generate_explanation_text(top_contributions, prediction)
            }
            
            return explanation
            
        except Exception as e:
            print(f"Error explaining prediction: {e}")
            return {}
    
    def _generate_explanation_text(self, contributions: pd.DataFrame, prediction: int) -> str:
        """Generate human-readable explanation text."""
        prediction_text = "churn" if prediction == 1 else "not churn"
        
        # Get top positive and negative contributions
        positive_contrib = contributions[contributions['shap_value'] > 0].head(3)
        negative_contrib = contributions[contributions['shap_value'] < 0].head(3)
        
        explanation = f"The model predicts this customer will {prediction_text}. "
        
        if not positive_contrib.empty:
            explanation += "Key factors increasing churn probability: "
            for _, row in positive_contrib.iterrows():
                explanation += f"{row['feature']} (impact: +{row['shap_value']:.3f}), "
            explanation = explanation.rstrip(", ") + ". "
        
        if not negative_contrib.empty:
            explanation += "Key factors decreasing churn probability: "
            for _, row in negative_contrib.iterrows():
                explanation += f"{row['feature']} (impact: {row['shap_value']:.3f}), "
            explanation = explanation.rstrip(", ") + "."
        
        return explanation


def analyze_model_interpretability(models: Dict[str, Pipeline], X_train: pd.DataFrame,
                                 X_test: pd.DataFrame, model_names: Optional[List[str]] = None) -> ModelInterpreter:
    """
    Perform comprehensive interpretability analysis for models.
    
    Args:
        models (Dict[str, Pipeline]): Dictionary of trained models
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Test data
        model_names (List[str], optional): Specific models to analyze
        
    Returns:
        ModelInterpreter: Configured interpreter with SHAP values
    """
    if not SHAP_AVAILABLE:
        print("SHAP is not available. Please install it to perform interpretability analysis.")
        return ModelInterpreter()
    
    print("=== MODEL INTERPRETABILITY ANALYSIS ===")
    
    interpreter = ModelInterpreter()
    
    # Select models to analyze
    if model_names is None:
        model_names = list(models.keys())
    
    # Limit to tree-based and linear models for better SHAP support
    supported_models = []
    for name in model_names:
        if name in models:
            classifier_type = type(models[name].named_steps['classifier']).__name__
            if any(keyword in classifier_type.lower() for keyword in 
                  ['tree', 'forest', 'xgb', 'gradient', 'linear', 'logistic']):
                supported_models.append(name)
            else:
                print(f"Skipping {name} - limited SHAP support for {classifier_type}")
    
    # Create explainers and calculate SHAP values
    for model_name in supported_models[:3]:  # Limit to top 3 for performance
        model = models[model_name]
        
        # Create explainer
        interpreter.create_explainer(model, X_train, model_name)
        
        # Calculate SHAP values
        if model_name in interpreter.explainers:
            interpreter.calculate_shap_values(model, X_test, model_name, max_samples=500)
    
    return interpreter