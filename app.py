#!/usr/bin/env python3
"""
Customer Churn Prediction Web Application

A comprehensive Streamlit dashboard for predicting customer churn with 
advanced analytics, model performance metrics, and interpretability features.

Features:
- Interactive prediction interface
- Real-time churn probability calculation  
- Model performance analysis
- Data insights and correlation analysis
- Batch prediction capability
- Model comparison dashboard
- SHAP explainability

Author: Customer Churn Prediction Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import project modules
from webutils.web_utils import (
    load_models, get_model_list, load_and_cache_data,
    set_page_config, load_custom_css, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
)

def main():
    """Main application entry point."""
    
    # Set page configuration
    set_page_config()
    
    # Load custom CSS
    load_custom_css()
    
    # Sidebar for navigation and model selection
    with st.sidebar:
        st.title("🎯 Churn Flow")
        st.markdown("---")
        
        # Model selection
        available_models = get_model_list()
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=0 if available_models else None,
            help="Choose the machine learning model for predictions"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        st.markdown("""
        Use the pages in the sidebar to:
        - **🔍 Insights**: Explore data patterns 
        - **📊 Performance**: View model metrics
        - **⚖️ Compare**: Compare different models
        - **🎯 Prediction**: Make individual predictions
        - **📁 Batch**: Upload CSV for bulk predictions
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses machine learning to predict customer churn
        based on demographic, account, and service usage data.
        
        **Risk Categories:**
        - 🟢 Low Risk: 0-30%
        - 🟡 Medium Risk: 30-70%  
        - 🔴 High Risk: 70%+
        """)
    
    # Main content area
    st.title("Churn Flow Application 🏠")
    st.markdown("---")
    st.title("Customer Churn Prediction Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Available Models", 
            len(available_models),
            help="Number of trained models available"
        )
    
    with col2:
        # Load sample data to get feature count
        try:
            sample_data = load_and_cache_data()
            feature_count = len(CATEGORICAL_FEATURES) + len(NUMERICAL_FEATURES)
            st.metric(
                "Input Features",
                feature_count,
                help="Number of customer attributes used for prediction"
            )
        except Exception as e:
            st.metric("Input Features", "N/A")
    
    with col3:
        st.metric(
            "Current Model",
            selected_model.replace('_', ' ').title() if selected_model else "None",
            help="Currently selected prediction model"
        )
    
    st.markdown("---")
    
    # Welcome message and instructions
    st.markdown("""
    ## Welcome to the Customer Churn Prediction System! 👋
    
    This comprehensive dashboard helps you predict and analyze customer churn using advanced machine learning models.
    
    ### 🚀 Getting Started
    1. **🔍 Explore Insights**: Discover patterns in customer data and feature importance
    2. **📊 View Performance**: Check model accuracy and performance metrics on the Performance page
    3. **⚖️ Compare Models**: Evaluate different algorithms side by side
    4. **🎯 Make a Prediction**: Navigate to the Prediction page to input customer data and get instant churn probability
    5. **📁 Batch Processing**: Upload CSV files for bulk predictions

    ### 📈 Key Features
    
    - **Interactive Visualizations** for data exploration
    - **Real-time Predictions** with confidence scores
    - **Risk Assessment** with color-coded results  
    - **Model Explainability** using SHAP values
    - **Business Impact Analysis** with revenue calculations
    - **Professional Export** options for results
    """)
    
    # Feature overview
    with st.expander("📋 Available Input Features"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Categorical Features:**")
            for feature in CATEGORICAL_FEATURES:
                st.markdown(f"• {feature}")
        
        with col2:
            st.markdown("**🔢 Numerical Features:**") 
            for feature in NUMERICAL_FEATURES:
                st.markdown(f"• {feature}")
    
    # Quick stats
    if selected_model:
        try:
            models = load_models()
            if selected_model in models:
                model = models[selected_model]
                st.success(f"✅ Model '{selected_model}' loaded successfully!")
                
                # Show model info
                with st.expander("🔧 Model Information"):
                    st.write(f"**Type:** {type(model).__name__}")
                    if hasattr(model, 'named_steps'):
                        st.write(f"**Pipeline Steps:** {list(model.named_steps.keys())}")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666666;'>
        <p>Built with ❤️ using Streamlit • Customer Churn Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()