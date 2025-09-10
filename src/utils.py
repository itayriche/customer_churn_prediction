"""Utility functions for customer churn prediction project."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from .config import FIGURE_SIZE, COLORS, TARGET_COLUMN


def setup_plotting_style() -> None:
    """Set up consistent plotting style for all visualizations."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def create_churn_distribution_plot(df: pd.DataFrame, target_col: str = TARGET_COLUMN,
                                 figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Create a bar plot showing the distribution of churn vs no-churn.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
        figsize (Tuple[int, int]): Figure size
    """
    plt.figure(figsize=figsize)
    
    churn_counts = df[target_col].value_counts()
    bars = plt.bar(churn_counts.index, churn_counts.values, color=COLORS)
    
    plt.xlabel('Churn Status')
    plt.ylabel('Count')
    plt.title('Distribution of Customer Churn')
    
    # Add value labels on bars
    for bar, count in zip(bars, churn_counts.values):
        height = bar.get_height()
        percentage = (count / churn_counts.sum()) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def create_numerical_histograms(df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Create histograms for all numerical features.
    
    Args:
        df (pd.DataFrame): Dataset
        figsize (Tuple[int, int]): Figure size
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numerical_cols:
        print("No numerical columns found.")
        return
    
    plt.figure(figsize=figsize)
    df[numerical_cols].hist(bins=30, figsize=figsize, layout=(len(numerical_cols)//3 + 1, 3))
    plt.suptitle('Distribution of Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.show()


def create_numerical_churn_comparison(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> None:
    """
    Create histograms comparing numerical features between churn and no-churn customers.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for column in numerical_cols:
        plt.figure(figsize=(10, 6))
        
        for churn_value in df[target_col].unique():
            subset = df[df[target_col] == churn_value][column]
            plt.hist(subset, bins=30, alpha=0.7, label=f'{target_col}: {churn_value}', density=True)
        
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.title(f'Distribution of {column} by Churn Status')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def create_categorical_churn_analysis(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> None:
    """
    Create bar plots showing churn rates for each categorical feature.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target column from categorical columns
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    for column in categorical_cols:
        plt.figure(figsize=(10, 6))
        
        # Calculate churn rates for each category
        churn_data = df.groupby(column)[target_col].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).reset_index()
        churn_data.columns = [column, 'churn_rate']
        
        # Create bar plot
        bars = plt.bar(churn_data[column], churn_data['churn_rate'])
        
        plt.xlabel(column)
        plt.ylabel('Churn Rate (%)')
        plt.title(f'Churn Rate by {column}')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, rate in zip(bars, churn_data['churn_rate']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()


def create_correlation_heatmap(df: pd.DataFrame, target_col: str = TARGET_COLUMN,
                             figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Create a correlation heatmap for all numerical features and encoded categorical features.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
        figsize (Tuple[int, int]): Figure size
    """
    # Encode categorical variables for correlation calculation
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    plt.figure(figsize=figsize)
    correlation_matrix = df_encoded.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()


def display_correlation_with_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN,
                                  top_n: int = 15) -> pd.DataFrame:
    """
    Display features most correlated with the target variable.
    
    Args:
        df (pd.DataFrame): Dataset
        target_col (str): Name of target column
        top_n (int): Number of top correlations to show
        
    Returns:
        pd.DataFrame: Correlation values sorted by absolute value
    """
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Find the target column in encoded data
    target_encoded = None
    for col in df_encoded.columns:
        if target_col in col:
            target_encoded = col
            break
    
    if target_encoded is None:
        # Target might already be numeric
        target_numeric = df[target_col].replace({'No': 0, 'Yes': 1})
        features_encoded = df_encoded.drop([col for col in df_encoded.columns if target_col in col], axis=1)
        correlations = features_encoded.corrwith(target_numeric)
    else:
        correlations = df_encoded.corr()[target_encoded]
        correlations = correlations.drop(target_encoded)
    
    # Sort by absolute correlation
    correlations_abs = correlations.abs().sort_values(ascending=False)
    correlations_sorted = correlations[correlations_abs.index]
    
    print(f"Top {top_n} Features Correlated with {target_col}:")
    print("=" * 50)
    
    correlation_df = pd.DataFrame({
        'Feature': correlations_sorted.head(top_n).index,
        'Correlation': correlations_sorted.head(top_n).values
    })
    
    print(correlation_df.to_string(index=False))
    
    return correlation_df


def save_dataframe_summary(df: pd.DataFrame, filename: str = None) -> str:
    """
    Save a comprehensive summary of the dataframe to a text file.
    
    Args:
        df (pd.DataFrame): Dataset to summarize
        filename (str, optional): Output filename. If None, generates timestamp-based name.
        
    Returns:
        str: Path to saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_summary_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("DATASET SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("BASIC INFORMATION:\n")
        f.write(f"Shape: {df.shape}\n")
        f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        f.write("COLUMN INFORMATION:\n")
        f.write(str(df.info(buf=None)) + "\n\n")
        
        f.write("DESCRIPTIVE STATISTICS:\n")
        f.write(str(df.describe(include='all')) + "\n\n")
        
        f.write("MISSING VALUES:\n")
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            f.write("No missing values found.\n")
        else:
            f.write(str(missing_values[missing_values > 0]) + "\n")
        
        f.write("\nUNIQUE VALUES PER COLUMN:\n")
        for col in df.columns:
            unique_count = df[col].nunique()
            f.write(f"{col}: {unique_count} unique values\n")
    
    print(f"Data summary saved to: {filename}")
    return filename


def create_model_comparison_table(results: Dict[str, Dict[str, Any]],
                                metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']) -> pd.DataFrame:
    """
    Create a formatted comparison table of model performance.
    
    Args:
        results (Dict[str, Dict[str, Any]]): Model evaluation results
        metrics (List[str]): Metrics to include in comparison
        
    Returns:
        pd.DataFrame: Formatted comparison table
    """
    comparison_data = []
    
    for model_name, model_results in results.items():
        if 'metrics' in model_results:
            row = {'Model': model_name}
            for metric in metrics:
                if metric in model_results['metrics']:
                    row[metric] = model_results['metrics'][metric]
                else:
                    row[metric] = np.nan
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Format numerical columns
    for col in df.columns:
        if col != 'Model' and df[col].dtype in ['float64', 'float32']:
            df[col] = df[col].round(4)
    
    # Sort by ROC AUC if available
    if 'roc_auc' in df.columns:
        df = df.sort_values('roc_auc', ascending=False)
    
    return df


def save_model_with_metadata(model: Any, model_name: str, metadata: Dict[str, Any],
                           save_dir: str = "models/") -> str:
    """
    Save a model with associated metadata.
    
    Args:
        model: Trained model to save
        model_name (str): Name of the model
        metadata (Dict[str, Any]): Model metadata (performance, parameters, etc.)
        save_dir (str): Directory to save the model
        
    Returns:
        str: Path to saved model file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata_filename = f"{model_name}_{timestamp}_metadata.joblib"
    metadata_path = os.path.join(save_dir, metadata_filename)
    joblib.dump(metadata, metadata_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    
    return model_path


def load_model_with_metadata(model_path: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a model and its associated metadata.
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        Tuple[Any, Dict[str, Any]]: Loaded model and metadata
    """
    # Load model
    model = joblib.load(model_path)
    
    # Construct metadata path
    metadata_path = model_path.replace('.joblib', '_metadata.joblib')
    
    metadata = {}
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
    else:
        print(f"Warning: Metadata file not found: {metadata_path}")
    
    return model, metadata


def calculate_business_impact(evaluation_results: Dict[str, Dict[str, Any]],
                            customer_value: float = 1000,
                            retention_cost: float = 100) -> pd.DataFrame:
    """
    Calculate business impact metrics for churn prediction models.
    
    Args:
        evaluation_results (Dict[str, Dict[str, Any]]): Model evaluation results
        customer_value (float): Average customer lifetime value
        retention_cost (float): Cost of retention campaign per customer
        
    Returns:
        pd.DataFrame: Business impact analysis
    """
    business_metrics = []
    
    for model_name, results in evaluation_results.items():
        if 'confusion_matrix' not in results:
            continue
        
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate business metrics
        total_customers = tn + fp + fn + tp
        total_churners = fn + tp
        
        # Customers saved (true positives)
        customers_saved = tp
        
        # Cost of false positives (unnecessary retention campaigns)
        false_positive_cost = fp * retention_cost
        
        # Revenue saved (customers correctly identified as churners)
        revenue_saved = customers_saved * customer_value
        
        # Cost of retention campaigns for true positives
        retention_campaign_cost = tp * retention_cost
        
        # Net benefit
        net_benefit = revenue_saved - retention_campaign_cost - false_positive_cost
        
        # Lost revenue from false negatives (missed churners)
        lost_revenue = fn * customer_value
        
        business_metrics.append({
            'Model': model_name,
            'Customers_Saved': customers_saved,
            'Revenue_Saved': revenue_saved,
            'Campaign_Cost': retention_campaign_cost + false_positive_cost,
            'Net_Benefit': net_benefit,
            'Lost_Revenue': lost_revenue,
            'ROI': (net_benefit / (retention_campaign_cost + false_positive_cost)) * 100 if (retention_campaign_cost + false_positive_cost) > 0 else 0
        })
    
    business_df = pd.DataFrame(business_metrics)
    return business_df.sort_values('Net_Benefit', ascending=False)


def print_section_header(title: str, width: int = 60, char: str = "=") -> None:
    """
    Print a formatted section header.
    
    Args:
        title (str): Section title
        width (int): Total width of header
        char (str): Character to use for decoration
    """
    print()
    print(char * width)
    print(f" {title.upper()}")
    print(char * width)


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"