"""
Model Performance Dashboard

This page displays comprehensive performance metrics for all trained models
including accuracy, precision, recall, ROC curves, and confusion matrices.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE, TEST_SIZE
from utils.web_utils import (
    load_models, load_and_cache_data, load_custom_css,
    display_model_metrics, create_confusion_matrix_heatmap,
    create_feature_importance_chart
)
from src.data_preprocessing import create_preprocessor
from sklearn.model_selection import train_test_split

@st.cache_data
def evaluate_all_models():
    """Evaluate all loaded models and return performance metrics."""
    try:
        # Load data
        df = load_and_cache_data()
        if df.empty:
            return {}
        
        # Prepare data
        X = df.drop('Churn', axis=1)
        y = df['Churn'].map({'No': 0, 'Yes': 1})
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        # Load models
        models = load_models()
        results = {}
        
        for model_name, model in models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'confusion_matrix': cm,
                    'fpr': fpr,
                    'tpr': tpr,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
            except Exception as e:
                st.warning(f"Could not evaluate {model_name}: {str(e)}")
                continue
        
        return results
        
    except Exception as e:
        st.error(f"Error evaluating models: {str(e)}")
        return {}

def create_roc_curves(results):
    """Create ROC curves comparison chart."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, (model_name, metrics) in enumerate(results.items()):
        if 'fpr' in metrics and 'tpr' in metrics:
            fig.add_trace(go.Scatter(
                x=metrics['fpr'],
                y=metrics['tpr'],
                mode='lines',
                name=f"{model_name.replace('_', ' ').title()} (AUC = {metrics['roc_auc']:.3f})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500,
        legend=dict(x=0.6, y=0.1)
    )
    
    return fig

def create_metrics_comparison(results):
    """Create a bar chart comparing model metrics."""
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=[m.replace('_', ' ').title() for m in metrics],
        specs=[[{"secondary_y": False}] * len(metrics)]
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric.title(),
                marker_color=colors[i % len(colors)],
                text=[f"{v:.3f}" for v in values],
                textposition='auto'
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title='Model Performance Comparison',
        height=400,
        showlegend=False
    )
    
    return fig

def create_cv_scores_plot(results):
    """Create cross-validation scores plot."""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, (model_name, metrics) in enumerate(results.items()):
        if 'cv_scores' in metrics:
            fig.add_trace(go.Box(
                y=metrics['cv_scores'],
                name=model_name.replace('_', ' ').title(),
                marker_color=colors[i % len(colors)]
            ))
    
    fig.update_layout(
        title='Cross-Validation Scores Distribution',
        yaxis_title='Accuracy Score',
        height=400
    )
    
    return fig

def display_feature_importance(model_name, model):
    """Display feature importance for tree-based models."""
    try:
        if hasattr(model, 'feature_importances_'):
            # Get feature names from the preprocessor
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                feature_names = []
                
                # Get categorical feature names
                if hasattr(preprocessor, 'named_transformers_'):
                    cat_transformer = preprocessor.named_transformers_.get('cat')
                    if hasattr(cat_transformer, 'get_feature_names_out'):
                        cat_features = cat_transformer.get_feature_names_out(CATEGORICAL_FEATURES)
                        feature_names.extend(cat_features)
                
                # Add numerical features
                feature_names.extend(NUMERICAL_FEATURES)
                
                # Get importance values
                importance_values = model.feature_importances_
                
                if len(feature_names) == len(importance_values):
                    fig = create_feature_importance_chart(
                        feature_names, 
                        importance_values,
                        f"Feature Importance - {model_name.replace('_', ' ').title()}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance chart not available due to dimension mismatch")
            else:
                st.info("Feature importance not available for this model type")
        else:
            st.info("This model type doesn't provide feature importance")
    except Exception as e:
        st.warning(f"Could not display feature importance: {str(e)}")

def main():
    load_custom_css()
    
    st.title("📊 Model Performance Dashboard")
    st.markdown("Comprehensive analysis of model performance metrics and comparisons.")
    
    # Load and evaluate models
    with st.spinner("Evaluating models..."):
        results = evaluate_all_models()
    
    if not results:
        st.error("No model evaluation results available. Please ensure models are trained.")
        return
    
    # Overview metrics
    st.subheader("📈 Performance Overview")
    
    # Create metrics dataframe for display
    metrics_df = pd.DataFrame({
        'Model': [name.replace('_', ' ').title() for name in results.keys()],
        'Accuracy': [results[name]['accuracy'] for name in results.keys()],
        'Precision': [results[name]['precision'] for name in results.keys()],
        'Recall': [results[name]['recall'] for name in results.keys()],
        'F1-Score': [results[name]['f1'] for name in results.keys()],
        'ROC AUC': [results[name]['roc_auc'] for name in results.keys()],
        'CV Mean': [results[name]['cv_mean'] for name in results.keys()],
        'CV Std': [results[name]['cv_std'] for name in results.keys()]
    })
    
    # Display metrics table
    st.dataframe(
        metrics_df.style.format({
            'Accuracy': '{:.3f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1-Score': '{:.3f}',
            'ROC AUC': '{:.3f}',
            'CV Mean': '{:.3f}',
            'CV Std': '{:.3f}'
        }).highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']),
        use_container_width=True
    )
    
    # Best model highlight
    best_model_idx = metrics_df['ROC AUC'].idxmax()
    best_model = metrics_df.iloc[best_model_idx]['Model']
    best_auc = metrics_df.iloc[best_model_idx]['ROC AUC']
    
    st.success(f"🏆 **Best Model:** {best_model} with ROC AUC of {best_auc:.3f}")
    
    # Performance visualizations
    st.subheader("📊 Performance Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curves
        st.plotly_chart(create_roc_curves(results), use_container_width=True)
    
    with col2:
        # Cross-validation scores
        st.plotly_chart(create_cv_scores_plot(results), use_container_width=True)
    
    # Metrics comparison
    st.plotly_chart(create_metrics_comparison(results), use_container_width=True)
    
    # Detailed model analysis
    st.subheader("🔍 Detailed Model Analysis")
    
    selected_model = st.selectbox(
        "Select Model for Detailed Analysis",
        options=list(results.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if selected_model in results:
        model_results = results[selected_model]
        
        # Display detailed metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{model_results['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{model_results['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{model_results['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{model_results['f1']:.3f}")
        with col5:
            st.metric("ROC AUC", f"{model_results['roc_auc']:.3f}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix
            cm_fig = create_confusion_matrix_heatmap(
                model_results['confusion_matrix'],
                ['No Churn', 'Churn']
            )
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with col2:
            # Classification report
            st.markdown("**Classification Report:**")
            report = classification_report(
                model_results['y_test'], 
                model_results['y_pred'],
                target_names=['No Churn', 'Churn'],
                output_dict=True
            )
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(
                report_df.style.format('{:.3f}'),
                use_container_width=True
            )
        
        # Feature importance (if available)
        st.subheader("🎯 Feature Importance")
        models = load_models()
        if selected_model in models:
            display_feature_importance(selected_model, models[selected_model])
        
        # Cross-validation details
        with st.expander("📊 Cross-Validation Details"):
            cv_scores = model_results['cv_scores']
            st.write(f"**Cross-Validation Scores:** {cv_scores}")
            st.write(f"**Mean:** {cv_scores.mean():.3f}")
            st.write(f"**Standard Deviation:** {cv_scores.std():.3f}")
            st.write(f"**95% Confidence Interval:** [{cv_scores.mean() - 1.96*cv_scores.std():.3f}, {cv_scores.mean() + 1.96*cv_scores.std():.3f}]")
    
    # Model recommendations
    st.subheader("💡 Model Recommendations")
    
    # Find best models for different metrics
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_precision = max(results.items(), key=lambda x: x[1]['precision'])
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **🎯 For Overall Performance:**
        - Use **{best_accuracy[0].replace('_', ' ').title()}** (Accuracy: {best_accuracy[1]['accuracy']:.3f})
        
        **🔍 For Precision (Low False Positives):**
        - Use **{best_precision[0].replace('_', ' ').title()}** (Precision: {best_precision[1]['precision']:.3f})
        """)
    
    with col2:
        st.info(f"""
        **🎣 For Recall (Catch More Churners):**
        - Use **{best_recall[0].replace('_', ' ').title()}** (Recall: {best_recall[1]['recall']:.3f})
        
        **⚖️ For Balanced Performance:**
        - Use **{best_f1[0].replace('_', ' ').title()}** (F1-Score: {best_f1[1]['f1']:.3f})
        """)

if __name__ == "__main__":
    main()