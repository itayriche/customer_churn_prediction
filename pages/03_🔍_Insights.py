"""
Data Insights and Analytics Dashboard

This page provides comprehensive data exploration, correlation analysis,
and insights into customer churn patterns and feature relationships.
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

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from webutils.web_utils import load_and_cache_data, load_custom_css, load_models

def create_churn_distribution():
    """Create churn distribution visualization."""
    df = load_and_cache_data()
    if df.empty:
        return None
    
    churn_counts = df['Churn'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=['No Churn', 'Churn'],
            y=[churn_counts['No'], churn_counts['Yes']],
            marker_color=['#2E86AB', '#A23B72'],
            text=[churn_counts['No'], churn_counts['Yes']],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Customer Churn Distribution',
        xaxis_title='Churn Status',
        yaxis_title='Number of Customers',
        height=400
    )
    
    return fig

def create_feature_correlation_heatmap():
    """Create correlation heatmap for numerical features."""
    df = load_and_cache_data()
    if df.empty:
        return None
    
    # Convert churn to numeric
    df_numeric = df.copy()
    df_numeric['Churn'] = df_numeric['Churn'].map({'No': 0, 'Yes': 1})
    
    # Select numerical features and target
    numeric_features = NUMERICAL_FEATURES + ['Churn']
    correlation_matrix = df_numeric[numeric_features].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        height=500,
        width=500
    )
    
    return fig

def create_categorical_churn_analysis():
    """Create analysis for categorical features vs churn."""
    df = load_and_cache_data()
    if df.empty:
        return None
    
    # Calculate churn rates for each categorical feature
    churn_rates = {}
    
    for feature in CATEGORICAL_FEATURES[:6]:  # Show first 6 features
        if feature in df.columns:
            churn_rate = df.groupby(feature)['Churn'].apply(
                lambda x: (x == 'Yes').sum() / len(x) * 100
            ).sort_values(ascending=False)
            churn_rates[feature] = churn_rate
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(churn_rates.keys()),
        specs=[[{"type": "bar"}] * 3] * 2
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, (feature, rates) in enumerate(churn_rates.items()):
        row = i // 3 + 1
        col = i % 3 + 1
        
        fig.add_trace(
            go.Bar(
                x=rates.index,
                y=rates.values,
                name=feature,
                marker_color=colors[i % len(colors)],
                text=[f"{v:.1f}%" for v in rates.values],
                textposition='auto'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Churn Rate by Categorical Features',
        height=600,
        showlegend=False
    )
    
    return fig

def create_numerical_feature_analysis():
    """Create analysis for numerical features."""
    df = load_and_cache_data()
    if df.empty:
        return None
    
    fig = make_subplots(
        rows=1, cols=len(NUMERICAL_FEATURES),
        subplot_titles=NUMERICAL_FEATURES,
        specs=[[{"type": "box"}] * len(NUMERICAL_FEATURES)]
    )
    
    colors = ['#FF6B6B', '#4ECDC4']
    
    for i, feature in enumerate(NUMERICAL_FEATURES):
        for j, churn_status in enumerate(['No', 'Yes']):
            data = df[df['Churn'] == churn_status][feature]
            
            fig.add_trace(
                go.Box(
                    y=data,
                    name=f"Churn: {churn_status}",
                    marker_color=colors[j],
                    showlegend=(i == 0)  # Only show legend for first subplot
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        title='Numerical Features Distribution by Churn Status',
        height=400
    )
    
    return fig

def create_customer_segments():
    """Create customer segmentation analysis."""
    df = load_and_cache_data()
    if df.empty:
        return None
    
    # Create segments based on tenure and monthly charges
    df_viz = df.copy()
    df_viz['TenureGroup'] = pd.cut(
        df_viz['tenure'], 
        bins=[0, 12, 24, 48, 100], 
        labels=['0-12 months', '12-24 months', '24-48 months', '48+ months']
    )
    df_viz['ChargesGroup'] = pd.cut(
        df_viz['MonthlyCharges'], 
        bins=[0, 35, 65, 100], 
        labels=['Low ($0-35)', 'Medium ($35-65)', 'High ($65+)']
    )
    
    # Calculate churn rates by segments
    segment_churn = df_viz.groupby(['TenureGroup', 'ChargesGroup'])['Churn'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).reset_index()
    
    fig = px.scatter(
        segment_churn,
        x='TenureGroup',
        y='ChargesGroup',
        size='Churn',
        color='Churn',
        title='Customer Churn Rate by Tenure and Monthly Charges Segments',
        color_continuous_scale='Reds',
        size_max=50
    )
    
    fig.update_layout(height=400)
    
    return fig

def calculate_business_insights():
    """Calculate key business insights."""
    df = load_and_cache_data()
    if df.empty:
        return {}
    
    insights = {}
    
    # Overall churn rate
    insights['overall_churn_rate'] = (df['Churn'] == 'Yes').mean() * 100
    
    # Average customer value
    insights['avg_monthly_charges'] = df['MonthlyCharges'].mean()
    insights['avg_total_charges'] = df['TotalCharges'].mean()
    
    # High-risk segments
    fiber_optic_churn = df[df['InternetService'] == 'Fiber optic']['Churn']
    insights['fiber_optic_churn_rate'] = (fiber_optic_churn == 'Yes').mean() * 100
    
    monthly_contract_churn = df[df['Contract'] == 'Month-to-month']['Churn']
    insights['monthly_contract_churn_rate'] = (monthly_contract_churn == 'Yes').mean() * 100
    
    # Customer lifetime value by churn
    churned_customers = df[df['Churn'] == 'Yes']
    retained_customers = df[df['Churn'] == 'No']
    
    insights['churned_avg_tenure'] = churned_customers['tenure'].mean()
    insights['retained_avg_tenure'] = retained_customers['tenure'].mean()
    
    insights['churned_avg_total'] = churned_customers['TotalCharges'].mean()
    insights['retained_avg_total'] = retained_customers['TotalCharges'].mean()
    
    # Revenue impact
    total_customers = len(df)
    churned_count = (df['Churn'] == 'Yes').sum()
    monthly_revenue_loss = churned_customers['MonthlyCharges'].sum()
    
    insights['total_customers'] = total_customers
    insights['churned_count'] = churned_count
    insights['monthly_revenue_loss'] = monthly_revenue_loss
    insights['annual_revenue_loss'] = monthly_revenue_loss * 12
    
    return insights

def main():
    load_custom_css()
    
    st.title("🔍 Data Insights & Analytics")
    st.markdown("Explore customer data patterns, correlations, and churn insights.")
    
    # Load data
    df = load_and_cache_data()
    if df.empty:
        st.error("No data available for analysis.")
        return
    
    # Key metrics overview
    st.subheader("📊 Key Metrics Overview")
    
    insights = calculate_business_insights()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Customers",
            f"{insights.get('total_customers', 0):,}",
            help="Total number of customers in dataset"
        )
    
    with col2:
        st.metric(
            "Overall Churn Rate",
            f"{insights.get('overall_churn_rate', 0):.1f}%",
            help="Percentage of customers who churned"
        )
    
    with col3:
        st.metric(
            "Avg Monthly Charges",
            f"${insights.get('avg_monthly_charges', 0):.2f}",
            help="Average monthly charges per customer"
        )
    
    with col4:
        st.metric(
            "Monthly Revenue Loss",
            f"${insights.get('monthly_revenue_loss', 0):,.2f}",
            help="Monthly revenue lost due to churn"
        )
    
    # Churn distribution
    st.subheader("📈 Churn Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        churn_fig = create_churn_distribution()
        if churn_fig:
            st.plotly_chart(churn_fig, use_container_width=True)
    
    with col2:
        st.markdown("**Key Insights:**")
        st.write(f"• {insights.get('churned_count', 0)} customers churned")
        st.write(f"• Annual revenue loss: ${insights.get('annual_revenue_loss', 0):,.2f}")
        st.write(f"• Retention rate: {100 - insights.get('overall_churn_rate', 0):.1f}%")
    
    # Feature correlation analysis
    st.subheader("🔗 Feature Correlations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        corr_fig = create_feature_correlation_heatmap()
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
    
    with col2:
        st.markdown("**Correlation Insights:**")
        
        # Calculate some specific correlations
        df_numeric = df.copy()
        df_numeric['Churn'] = df_numeric['Churn'].map({'No': 0, 'Yes': 1})
        
        correlations = df_numeric[NUMERICAL_FEATURES + ['Churn']].corr()['Churn'].drop('Churn').sort_values(key=abs, ascending=False)
        
        st.write("**Strongest correlations with churn:**")
        for feature, corr in correlations.items():
            direction = "positive" if corr > 0 else "negative"
            st.write(f"• **{feature}**: {corr:.3f} ({direction})")
    
    # Categorical analysis
    st.subheader("🏷️ Categorical Feature Analysis")
    
    cat_fig = create_categorical_churn_analysis()
    if cat_fig:
        st.plotly_chart(cat_fig, use_container_width=True)
    
    # High-risk segments
    with st.expander("⚠️ High-Risk Segments"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.error(f"""
            **Fiber Optic Customers:**
            - Churn Rate: {insights.get('fiber_optic_churn_rate', 0):.1f}%
            - Recommendation: Improve fiber optic service quality
            """)
        
        with col2:
            st.warning(f"""
            **Month-to-Month Contracts:**
            - Churn Rate: {insights.get('monthly_contract_churn_rate', 0):.1f}%
            - Recommendation: Incentivize longer contracts
            """)
    
    # Numerical features analysis
    st.subheader("🔢 Numerical Features Analysis")
    
    num_fig = create_numerical_feature_analysis()
    if num_fig:
        st.plotly_chart(num_fig, use_container_width=True)
    
    # Customer segmentation
    st.subheader("👥 Customer Segmentation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        segment_fig = create_customer_segments()
        if segment_fig:
            st.plotly_chart(segment_fig, use_container_width=True)
    
    with col2:
        st.markdown("**Segment Insights:**")
        st.write("• New customers (0-12 months) with high charges are at highest risk")
        st.write("• Long-term customers (48+ months) show lower churn rates")
        st.write("• Medium charge customers are most stable")
    
    # Business impact analysis
    st.subheader("💼 Business Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Customer Lifetime Value:**
        - Churned customers avg tenure: {insights.get('churned_avg_tenure', 0):.1f} months
        - Retained customers avg tenure: {insights.get('retained_avg_tenure', 0):.1f} months
        - Churned customers avg total: ${insights.get('churned_avg_total', 0):,.2f}
        """)
    
    with col2:
        st.success(f"""
        **Retention Opportunity:**
        - If we reduce churn by 10%: ${insights.get('annual_revenue_loss', 0) * 0.1:,.2f}/year saved
        - If we reduce churn by 25%: ${insights.get('annual_revenue_loss', 0) * 0.25:,.2f}/year saved
        - If we reduce churn by 50%: ${insights.get('annual_revenue_loss', 0) * 0.5:,.2f}/year saved
        """)
    
    # Data quality and completeness
    with st.expander("📋 Data Quality Summary"):
        st.write("**Dataset Information:**")
        st.write(f"• Shape: {df.shape}")
        st.write(f"• Missing values: {df.isnull().sum().sum()}")
        st.write(f"• Duplicate rows: {df.duplicated().sum()}")
        
        st.write("**Feature Types:**")
        st.write(f"• Categorical features: {len(CATEGORICAL_FEATURES)}")
        st.write(f"• Numerical features: {len(NUMERICAL_FEATURES)}")
        
        # Show data types
        st.write("**Data Types:**")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.write(f"• {dtype}: {count} columns")

if __name__ == "__main__":
    main()