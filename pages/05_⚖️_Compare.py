"""
Model Comparison Dashboard

This page provides side-by-side comparison of different machine learning models
for customer churn prediction, including performance metrics, feature importance,
and business impact analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE, TEST_SIZE
from utils.web_utils import (
    load_models, load_and_cache_data, load_custom_css,
    create_confusion_matrix_heatmap, format_currency
)
from sklearn.model_selection import train_test_split

@st.cache_data
def get_model_comparison_data():
    """Get comprehensive comparison data for all models."""
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
        comparison_data = {}
        
        for model_name, model in models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate comprehensive metrics
                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score, 
                    roc_auc_score, average_precision_score, cohen_kappa_score,
                    matthews_corrcoef
                )
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'avg_precision': average_precision_score(y_test, y_pred_proba),
                    'kappa': cohen_kappa_score(y_test, y_pred),
                    'mcc': matthews_corrcoef(y_test, y_pred)
                }
                
                # Business metrics
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                # Assuming average customer value calculations
                avg_customer_value = df['MonthlyCharges'].mean() * 12  # Annual value
                
                business_metrics = {
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn,
                    'customers_saved': tp,  # Correctly identified churners
                    'false_alarms': fp,     # Incorrectly flagged as churners
                    'missed_churners': fn,  # Churners we missed
                    'revenue_saved': tp * avg_customer_value,
                    'unnecessary_costs': fp * 50,  # Cost of unnecessary retention efforts
                    'lost_revenue': fn * avg_customer_value
                }
                
                # ROC curve data
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                
                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier'), 'feature_importances_'):
                    feature_importance = model.named_steps['classifier'].feature_importances_
                
                comparison_data[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'business_metrics': business_metrics,
                    'confusion_matrix': cm,
                    'fpr': fpr,
                    'tpr': tpr,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'feature_importance': feature_importance
                }
                
            except Exception as e:
                st.warning(f"Could not evaluate {model_name}: {str(e)}")
                continue
        
        return comparison_data
        
    except Exception as e:
        st.error(f"Error in model comparison: {str(e)}")
        return {}

def create_metrics_radar_chart(comparison_data, selected_models):
    """Create radar chart comparing model metrics."""
    if not comparison_data or not selected_models:
        return None
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, model_name in enumerate(selected_models):
        if model_name in comparison_data:
            model_metrics = comparison_data[model_name]['metrics']
            values = [model_metrics[metric] for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model_name.replace('_', ' ').title(),
                line_color=colors[i % len(colors)]
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Model Performance Comparison",
        height=500
    )
    
    return fig

def create_business_impact_comparison(comparison_data, selected_models):
    """Create business impact comparison chart."""
    if not comparison_data or not selected_models:
        return None
    
    models = []
    revenue_saved = []
    unnecessary_costs = []
    lost_revenue = []
    net_benefit = []
    
    for model_name in selected_models:
        if model_name in comparison_data:
            bm = comparison_data[model_name]['business_metrics']
            models.append(model_name.replace('_', ' ').title())
            revenue_saved.append(bm['revenue_saved'])
            unnecessary_costs.append(bm['unnecessary_costs'])
            lost_revenue.append(bm['lost_revenue'])
            net_benefit.append(bm['revenue_saved'] - bm['unnecessary_costs'] - bm['lost_revenue'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Revenue Saved',
        x=models,
        y=revenue_saved,
        marker_color='green'
    ))
    
    fig.add_trace(go.Bar(
        name='Unnecessary Costs',
        x=models,
        y=[-cost for cost in unnecessary_costs],  # Negative values
        marker_color='orange'
    ))
    
    fig.add_trace(go.Bar(
        name='Lost Revenue',
        x=models,
        y=[-revenue for revenue in lost_revenue],  # Negative values
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Business Impact Comparison',
        xaxis_title='Models',
        yaxis_title='Revenue Impact ($)',
        barmode='relative',
        height=500
    )
    
    return fig

def create_roc_comparison(comparison_data, selected_models):
    """Create ROC curves comparison."""
    if not comparison_data or not selected_models:
        return None
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, model_name in enumerate(selected_models):
        if model_name in comparison_data:
            data = comparison_data[model_name]
            auc_score = data['metrics']['roc_auc']
            
            fig.add_trace(go.Scatter(
                x=data['fpr'],
                y=data['tpr'],
                mode='lines',
                name=f"{model_name.replace('_', ' ').title()} (AUC = {auc_score:.3f})",
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
        height=500
    )
    
    return fig

def create_confusion_matrices_comparison(comparison_data, selected_models):
    """Create side-by-side confusion matrices."""
    if not comparison_data or not selected_models:
        return None
    
    n_models = len(selected_models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[model.replace('_', ' ').title() for model in selected_models],
        specs=[[{"type": "heatmap"}] * cols for _ in range(rows)]
    )
    
    for i, model_name in enumerate(selected_models):
        if model_name in comparison_data:
            cm = comparison_data[model_name]['confusion_matrix']
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=cm,
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 14},
                    showscale=(i == 0)  # Only show scale for first matrix
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title='Confusion Matrices Comparison',
        height=300 * rows
    )
    
    return fig

def create_feature_importance_comparison(comparison_data, selected_models):
    """Compare feature importance across models."""
    if not comparison_data or not selected_models:
        return None
    
    # Check which models have feature importance
    models_with_importance = []
    for model_name in selected_models:
        if (model_name in comparison_data and 
            comparison_data[model_name]['feature_importance'] is not None):
            models_with_importance.append(model_name)
    
    if not models_with_importance:
        return None
    
    # Get feature names (simplified)
    n_features = len(comparison_data[models_with_importance[0]]['feature_importance'])
    feature_names = [f"Feature_{i+1}" for i in range(min(10, n_features))]  # Top 10 features
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    
    for i, model_name in enumerate(models_with_importance):
        importance = comparison_data[model_name]['feature_importance'][:10]  # Top 10
        
        fig.add_trace(go.Bar(
            name=model_name.replace('_', ' ').title(),
            x=feature_names,
            y=importance,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title='Feature Importance Comparison (Top 10 Features)',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        barmode='group',
        height=500
    )
    
    return fig

def main():
    load_custom_css()
    
    st.title("⚖️ Model Comparison Dashboard")
    st.markdown("Compare different machine learning models side-by-side for comprehensive analysis.")
    
    # Load comparison data
    with st.spinner("Loading model comparison data..."):
        comparison_data = get_model_comparison_data()
    
    if not comparison_data:
        st.error("No model comparison data available. Please ensure models are trained.")
        return
    
    available_models = list(comparison_data.keys())
    
    # Model selection
    st.subheader("🎯 Select Models to Compare")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_models = st.multiselect(
            "Choose models for comparison",
            options=available_models,
            default=available_models[:3] if len(available_models) >= 3 else available_models,
            help="Select 2-5 models for optimal comparison"
        )
    
    with col2:
        comparison_type = st.selectbox(
            "Comparison Focus",
            ["Performance Metrics", "Business Impact", "Technical Details"],
            help="Choose the type of comparison to emphasize"
        )
    
    if not selected_models:
        st.warning("Please select at least one model for comparison.")
        return
    
    if len(selected_models) == 1:
        st.info("Select multiple models to see comparison charts.")
    
    # Quick overview table
    st.subheader("📊 Quick Comparison Overview")
    
    overview_data = []
    for model_name in selected_models:
        if model_name in comparison_data:
            metrics = comparison_data[model_name]['metrics']
            business = comparison_data[model_name]['business_metrics']
            
            overview_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1']:.3f}",
                'ROC AUC': f"{metrics['roc_auc']:.3f}",
                'Customers Saved': business['customers_saved'],
                'Revenue Saved': f"${business['revenue_saved']:,.0f}",
                'Net Benefit': f"${business['revenue_saved'] - business['unnecessary_costs'] - business['lost_revenue']:,.0f}"
            })
    
    overview_df = pd.DataFrame(overview_data)
    st.dataframe(overview_df, use_container_width=True)
    
    # Find and highlight best models
    if len(selected_models) > 1:
        best_accuracy_idx = overview_df['Accuracy'].astype(float).idxmax()
        best_f1_idx = overview_df['F1-Score'].astype(float).idxmax()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🏆 **Best Accuracy:** {overview_df.loc[best_accuracy_idx, 'Model']} ({overview_df.loc[best_accuracy_idx, 'Accuracy']})")
        
        with col2:
            st.success(f"🏆 **Best F1-Score:** {overview_df.loc[best_f1_idx, 'Model']} ({overview_df.loc[best_f1_idx, 'F1-Score']})")
    
    # Visualization section
    if len(selected_models) > 1:
        st.subheader("📈 Performance Visualizations")
        
        # Performance metrics radar chart
        radar_fig = create_metrics_radar_chart(comparison_data, selected_models)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ROC curves
            roc_fig = create_roc_comparison(comparison_data, selected_models)
            if roc_fig:
                st.plotly_chart(roc_fig, use_container_width=True)
        
        with col2:
            # Business impact
            business_fig = create_business_impact_comparison(comparison_data, selected_models)
            if business_fig:
                st.plotly_chart(business_fig, use_container_width=True)
    
    # Detailed analysis section
    st.subheader("🔍 Detailed Analysis")
    
    if comparison_type == "Performance Metrics":
        # Confusion matrices
        cm_fig = create_confusion_matrices_comparison(comparison_data, selected_models)
        if cm_fig:
            st.plotly_chart(cm_fig, use_container_width=True)
        
        # Additional metrics table
        detailed_metrics = []
        for model_name in selected_models:
            if model_name in comparison_data:
                metrics = comparison_data[model_name]['metrics']
                detailed_metrics.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1'],
                    'ROC AUC': metrics['roc_auc'],
                    'Avg Precision': metrics['avg_precision'],
                    'Cohen\'s Kappa': metrics['kappa'],
                    'Matthews Corr': metrics['mcc']
                })
        
        detailed_df = pd.DataFrame(detailed_metrics)
        st.dataframe(
            detailed_df.style.format({
                col: '{:.3f}' for col in detailed_df.columns if col != 'Model'
            }).highlight_max(axis=0, subset=[col for col in detailed_df.columns if col != 'Model']),
            use_container_width=True
        )
    
    elif comparison_type == "Business Impact":
        # Business metrics table
        business_data = []
        for model_name in selected_models:
            if model_name in comparison_data:
                bm = comparison_data[model_name]['business_metrics']
                net_benefit = bm['revenue_saved'] - bm['unnecessary_costs'] - bm['lost_revenue']
                roi = (net_benefit / (bm['unnecessary_costs'] + 1)) * 100  # Avoid division by zero
                
                business_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Customers Saved': bm['customers_saved'],
                    'False Alarms': bm['false_alarms'],
                    'Missed Churners': bm['missed_churners'],
                    'Revenue Saved': bm['revenue_saved'],
                    'Unnecessary Costs': bm['unnecessary_costs'],
                    'Lost Revenue': bm['lost_revenue'],
                    'Net Benefit': net_benefit,
                    'ROI (%)': roi
                })
        
        business_df = pd.DataFrame(business_data)
        
        # Format currency columns
        currency_cols = ['Revenue Saved', 'Unnecessary Costs', 'Lost Revenue', 'Net Benefit']
        for col in currency_cols:
            business_df[col] = business_df[col].apply(lambda x: f"${x:,.0f}")
        
        business_df['ROI (%)'] = business_df['ROI (%)'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(business_df, use_container_width=True)
        
        # Business recommendations
        st.subheader("💡 Business Recommendations")
        
        # Find best models for different business objectives
        raw_business_data = []
        for model_name in selected_models:
            if model_name in comparison_data:
                bm = comparison_data[model_name]['business_metrics']
                net_benefit = bm['revenue_saved'] - bm['unnecessary_costs'] - bm['lost_revenue']
                raw_business_data.append((model_name, bm, net_benefit))
        
        if raw_business_data:
            best_revenue = max(raw_business_data, key=lambda x: x[1]['revenue_saved'])
            best_net = max(raw_business_data, key=lambda x: x[2])
            least_false_alarms = min(raw_business_data, key=lambda x: x[1]['false_alarms'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"""
                **💰 Maximum Revenue Protection:**
                - Use **{best_revenue[0].replace('_', ' ').title()}**
                - Saves ${best_revenue[1]['revenue_saved']:,.0f} annually
                - Identifies {best_revenue[1]['customers_saved']} at-risk customers
                """)
                
                st.info(f"""
                **🎯 Minimal False Alarms:**
                - Use **{least_false_alarms[0].replace('_', ' ').title()}**
                - Only {least_false_alarms[1]['false_alarms']} false positives
                - Reduces unnecessary retention costs
                """)
            
            with col2:
                st.success(f"""
                **⚖️ Best Overall ROI:**
                - Use **{best_net[0].replace('_', ' ').title()}**
                - Net benefit: ${best_net[2]:,.0f}
                - Optimal balance of accuracy and cost
                """)
    
    elif comparison_type == "Technical Details":
        # Feature importance comparison
        importance_fig = create_feature_importance_comparison(comparison_data, selected_models)
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
        else:
            st.info("Feature importance comparison not available for selected models.")
        
        # Model characteristics
        st.subheader("🔧 Model Characteristics")
        
        for model_name in selected_models:
            if model_name in comparison_data:
                model = comparison_data[model_name]['model']
                
                with st.expander(f"📋 {model_name.replace('_', ' ').title()} Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Model Type:** {type(model).__name__}")
                        if hasattr(model, 'named_steps'):
                            st.write(f"**Pipeline Steps:** {list(model.named_steps.keys())}")
                        
                        # Model parameters (if accessible)
                        try:
                            if hasattr(model, 'get_params'):
                                params = model.get_params()
                                st.write("**Key Parameters:**")
                                # Show only non-default parameters
                                for key, value in list(params.items())[:5]:  # Limit display
                                    st.write(f"• {key}: {value}")
                        except:
                            pass
                    
                    with col2:
                        # Model performance summary
                        metrics = comparison_data[model_name]['metrics']
                        st.write("**Performance Summary:**")
                        st.write(f"• Accuracy: {metrics['accuracy']:.3f}")
                        st.write(f"• Precision: {metrics['precision']:.3f}")
                        st.write(f"• Recall: {metrics['recall']:.3f}")
                        st.write(f"• F1-Score: {metrics['f1']:.3f}")
                        st.write(f"• ROC AUC: {metrics['roc_auc']:.3f}")
    
    # Model selection recommendations
    st.subheader("🎯 Model Selection Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Choose based on your priority:**
        
        **For Maximum Accuracy:**
        - Use the model with highest overall accuracy
        - Best for general performance
        
        **For Catching Churners (High Recall):**
        - Use model with highest recall score  
        - Minimizes missed churn cases
        
        **For Precision Marketing:**
        - Use model with highest precision
        - Reduces false positive retention campaigns
        """)
    
    with col2:
        st.markdown("""
        **💼 Business Considerations:**
        
        **For Cost-Sensitive Applications:**
        - Balance false positives vs false negatives
        - Consider retention campaign costs
        
        **For Revenue Protection:**
        - Prioritize models that save most revenue
        - Focus on high-value customer retention
        
        **For Operational Efficiency:**
        - Choose models with best net benefit
        - Balance accuracy with operational costs
        """)

if __name__ == "__main__":
    main()