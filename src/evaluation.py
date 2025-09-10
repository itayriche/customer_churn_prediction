"""Model evaluation module for customer churn prediction."""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

from .config import FIGURE_SIZE, COLORS


class ModelEvaluator:
    """Class for evaluating machine learning models."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.evaluation_results = {}
        
    def evaluate_single_model(self, model: Pipeline, X_test: pd.DataFrame, 
                            y_test: pd.Series, model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model and return comprehensive metrics.
        
        Args:
            model (Pipeline): Trained model pipeline
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_name (str): Name of the model
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.evaluation_results[model_name] = results
        
        return results
    
    def evaluate_multiple_models(self, models: Dict[str, Pipeline], 
                               X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate multiple models and return comparison results.
        
        Args:
            models (Dict[str, Pipeline]): Dictionary of trained models
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            pd.DataFrame: Comparison of model performance
        """
        print("Evaluating models...")
        results_list = []
        
        for name, model in models.items():
            try:
                print(f"Evaluating {name}...")
                results = self.evaluate_single_model(model, X_test, y_test, name)
                
                # Extract metrics for comparison
                model_results = {'Model': name}
                model_results.update(results['metrics'])
                results_list.append(model_results)
                
                print(f"✓ {name} evaluated successfully")
                
            except Exception as e:
                print(f"✗ Error evaluating {name}: {e}")
        
        comparison_df = pd.DataFrame(results_list)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        return comparison_df
    
    def plot_confusion_matrices(self, figsize: Tuple[int, int] = FIGURE_SIZE) -> None:
        """
        Plot confusion matrices for all evaluated models.
        
        Args:
            figsize (Tuple[int, int]): Figure size
        """
        n_models = len(self.evaluation_results)
        if n_models == 0:
            print("No models to plot. Please evaluate models first.")
            return
        
        # Calculate grid dimensions
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (name, results) in enumerate(self.evaluation_results.items()):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name}\nAccuracy: {results["metrics"]["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_models, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series,
                       models: Dict[str, Pipeline], figsize: Tuple[int, int] = FIGURE_SIZE) -> None:
        """
        Plot ROC curves for multiple models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            models (Dict[str, Pipeline]): Dictionary of trained models
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        
        for name, model in models.items():
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    # Use decision function for SVM
                    y_pred_proba = model.decision_function(X_test)
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Plot
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
                
            except Exception as e:
                print(f"Error plotting ROC for {name}: {e}")
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_precision_recall_curves(self, X_test: pd.DataFrame, y_test: pd.Series,
                                   models: Dict[str, Pipeline], 
                                   figsize: Tuple[int, int] = FIGURE_SIZE) -> None:
        """
        Plot Precision-Recall curves for multiple models.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            models (Dict[str, Pipeline]): Dictionary of trained models
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        
        for name, model in models.items():
            try:
                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_pred_proba = model.decision_function(X_test)
                
                # Calculate Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                avg_precision = np.trapz(precision, recall)
                
                # Plot
                plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.2f})')
                
            except Exception as e:
                print(f"Error plotting PR curve for {name}: {e}")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                              model_name: str, top_n: int = 15,
                              figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot feature importance for a model.
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            model_name (str): Name of the model
            top_n (int): Number of top features to show
            figsize (Tuple[int, int]): Figure size
        """
        if importance_df is None or importance_df.empty:
            print(f"No feature importance data available for {model_name}")
            return
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=figsize)
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                            figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot comparison of multiple metrics across models.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
            metrics (List[str]): Metrics to compare
            figsize (Tuple[int, int]): Figure size
        """
        if comparison_df.empty:
            print("No comparison data available.")
            return
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=figsize)
        if len(available_metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Create bar plot
            bars = ax.bar(range(len(comparison_df)), comparison_df[metric])
            ax.set_xticks(range(len(comparison_df)))
            ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'Model Comparison - {metric.capitalize()}')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def generate_model_report(self, model_name: str) -> str:
        """
        Generate a detailed text report for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            str: Detailed model report
        """
        if model_name not in self.evaluation_results:
            return f"No evaluation results found for {model_name}"
        
        results = self.evaluation_results[model_name]
        metrics = results['metrics']
        class_report = results['classification_report']
        
        report = f"""
=== MODEL EVALUATION REPORT: {model_name.upper()} ===

PERFORMANCE METRICS:
• Accuracy:  {metrics['accuracy']:.4f}
• Precision: {metrics['precision']:.4f}
• Recall:    {metrics['recall']:.4f}
• F1-Score:  {metrics['f1']:.4f}
"""
        
        if 'roc_auc' in metrics:
            report += f"• ROC AUC:   {metrics['roc_auc']:.4f}\n"
        
        report += f"""
CONFUSION MATRIX:
{results['confusion_matrix']}

CLASSIFICATION REPORT:
              precision    recall  f1-score   support
         No      {class_report['0']['precision']:.2f}      {class_report['0']['recall']:.2f}      {class_report['0']['f1-score']:.2f}     {class_report['0']['support']}
        Yes      {class_report['1']['precision']:.2f}      {class_report['1']['recall']:.2f}      {class_report['1']['f1-score']:.2f}     {class_report['1']['support']}

   accuracy                          {class_report['accuracy']:.2f}     {class_report['macro avg']['support']}
  macro avg      {class_report['macro avg']['precision']:.2f}      {class_report['macro avg']['recall']:.2f}      {class_report['macro avg']['f1-score']:.2f}     {class_report['macro avg']['support']}
weighted avg      {class_report['weighted avg']['precision']:.2f}      {class_report['weighted avg']['recall']:.2f}      {class_report['weighted avg']['f1-score']:.2f}     {class_report['weighted avg']['support']}
"""
        
        return report
    
    def get_best_model(self, metric: str = 'roc_auc') -> Optional[str]:
        """
        Get the name of the best performing model based on a specific metric.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            str: Name of the best model or None
        """
        if not self.evaluation_results:
            return None
        
        best_model = None
        best_score = -1
        
        for name, results in self.evaluation_results.items():
            if metric in results['metrics']:
                score = results['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model = name
        
        return best_model


def evaluate_models_comprehensive(models: Dict[str, Pipeline], 
                                X_test: pd.DataFrame, y_test: pd.Series,
                                show_plots: bool = True) -> Tuple[ModelEvaluator, pd.DataFrame]:
    """
    Perform comprehensive evaluation of multiple models.
    
    Args:
        models (Dict[str, Pipeline]): Dictionary of trained models
        X_test (pd.DataFrame): Test features  
        y_test (pd.Series): Test target
        show_plots (bool): Whether to show plots
        
    Returns:
        Tuple[ModelEvaluator, pd.DataFrame]: Evaluator object and comparison dataframe
    """
    print("=== COMPREHENSIVE MODEL EVALUATION ===")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    comparison_df = evaluator.evaluate_multiple_models(models, X_test, y_test)
    
    if not comparison_df.empty:
        print("\nModel Performance Comparison:")
        print(comparison_df.round(4))
        
        # Get best model
        best_model = evaluator.get_best_model('roc_auc')
        if best_model:
            print(f"\nBest performing model (ROC AUC): {best_model}")
    
    if show_plots:
        # Plot comparisons
        evaluator.plot_model_comparison(comparison_df)
        evaluator.plot_roc_curves(X_test, y_test, models)
        evaluator.plot_precision_recall_curves(X_test, y_test, models)
        evaluator.plot_confusion_matrices()
    
    return evaluator, comparison_df