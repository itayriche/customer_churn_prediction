"""
Batch Prediction Interface

This page allows users to upload CSV files with customer data
for bulk churn predictions and download results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
import os
from pathlib import Path
from datetime import datetime
import base64

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from webutils.web_utils import (
    load_models, load_custom_css, get_risk_category, 
    format_probability, validate_input_data, show_loading_spinner
)

def create_sample_csv():
    """Create a sample CSV for download."""
    sample_data = {
        'customerID': ['7590-VHVEG', '5575-GNVDE', '3668-QPYBK'],
        'gender': ['Female', 'Male', 'Male'],
        'SeniorCitizen': ['No', 'No', 'No'],
        'Partner': ['Yes', 'No', 'No'],
        'Dependents': ['No', 'No', 'No'],
        'tenure': [1, 34, 2],
        'PhoneService': ['No', 'Yes', 'Yes'],
        'MultipleLines': ['No phone service', 'No', 'No'],
        'InternetService': ['DSL', 'DSL', 'DSL'],
        'OnlineSecurity': ['No', 'Yes', 'Yes'],
        'OnlineBackup': ['Yes', 'No', 'Yes'],
        'DeviceProtection': ['No', 'Yes', 'No'],
        'TechSupport': ['No', 'No', 'No'],
        'StreamingTV': ['No', 'No', 'No'],
        'StreamingMovies': ['No', 'No', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check'],
        'MonthlyCharges': [29.85, 56.95, 53.85],
        'TotalCharges': [29.85, 1889.50, 108.15]
    }
    
    return pd.DataFrame(sample_data)

def validate_csv_data(df):
    """Validate uploaded CSV data."""
    errors = []
    warnings = []
    
    # Check required columns
    required_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        errors.append(f"Missing required columns: {', '.join(missing_features)}")
    
    # Check for empty DataFrame
    if df.empty:
        errors.append("CSV file is empty")
        return False, errors, warnings
    
    # Check for missing values
    missing_counts = df[required_features].isnull().sum()
    missing_features = missing_counts[missing_counts > 0]
    
    if not missing_features.empty:
        for feature, count in missing_features.items():
            warnings.append(f"Column '{feature}' has {count} missing values")
    
    # Validate data types and ranges
    for feature in NUMERICAL_FEATURES:
        if feature in df.columns:
            try:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
            except:
                warnings.append(f"Could not convert {feature} to numeric")
    
    # Business logic validation
    if 'tenure' in df.columns:
        invalid_tenure = df[(df['tenure'] < 0) | (df['tenure'] > 100)]
        if not invalid_tenure.empty:
            warnings.append(f"{len(invalid_tenure)} rows have invalid tenure values")
    
    if 'MonthlyCharges' in df.columns:
        invalid_charges = df[(df['MonthlyCharges'] < 0) | (df['MonthlyCharges'] > 200)]
        if not invalid_charges.empty:
            warnings.append(f"{len(invalid_charges)} rows have invalid monthly charges")
    
    return len(errors) == 0, errors, warnings

def process_batch_predictions(df, model, model_name):
    """Process batch predictions for the uploaded data."""
    try:
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]
        
        # Create results dataframe
        results_df = df.copy()
        
        # Add prediction results
        results_df['Predicted_Churn'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]
        results_df['Churn_Probability'] = probabilities
        results_df['Probability_Percentage'] = [f"{p*100:.1f}%" for p in probabilities]
        
        # Add risk categories
        risk_info = [get_risk_category(p) for p in probabilities]
        results_df['Risk_Category'] = [info[0] for info in risk_info]
        results_df['Risk_Color'] = [info[1] for info in risk_info]
        
        # Add model info
        results_df['Model_Used'] = model_name
        results_df['Prediction_Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return results_df, None
        
    except Exception as e:
        return None, str(e)

def create_results_summary(results_df):
    """Create a summary of batch prediction results."""
    summary = {}
    
    # Overall statistics
    total_customers = len(results_df)
    churn_count = (results_df['Predicted_Churn'] == 'Churn').sum()
    no_churn_count = total_customers - churn_count
    
    summary['total_customers'] = total_customers
    summary['churn_count'] = churn_count
    summary['no_churn_count'] = no_churn_count
    summary['churn_rate'] = (churn_count / total_customers) * 100
    
    # Risk categories
    risk_counts = results_df['Risk_Category'].value_counts()
    summary['low_risk'] = risk_counts.get('Low Risk', 0)
    summary['medium_risk'] = risk_counts.get('Medium Risk', 0)
    summary['high_risk'] = risk_counts.get('High Risk', 0)
    
    # Financial impact
    avg_monthly_charges = results_df['MonthlyCharges'].mean()
    churning_customers = results_df[results_df['Predicted_Churn'] == 'Churn']
    potential_monthly_loss = churning_customers['MonthlyCharges'].sum()
    potential_annual_loss = potential_monthly_loss * 12
    
    summary['avg_monthly_charges'] = avg_monthly_charges
    summary['potential_monthly_loss'] = potential_monthly_loss
    summary['potential_annual_loss'] = potential_annual_loss
    
    return summary

def download_csv(df, filename):
    """Create download link for CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">📥 Download {filename}</a>'
    return href

def main():
    load_custom_css()
    
    st.title("📁 Batch Churn Prediction")
    st.markdown("Upload a CSV file with customer data to get bulk churn predictions.")
    
    # Load models
    models = load_models()
    if not models:
        st.error("No trained models found. Please train models first.")
        return
    
    # Model selection
    selected_model_name = st.selectbox(
        "Select Model for Batch Prediction",
        options=list(models.keys()),
        help="Choose which machine learning model to use for predictions"
    )
    
    selected_model = models[selected_model_name]
    
    # Instructions and sample file
    st.subheader("📋 Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📤 Upload Requirements:**
        1. CSV file format
        2. Include all required features
        3. Proper data types and formats
        4. No more than 10,000 rows recommended
        
        **Required Columns:**
        """)
        
        # Show required features in expandable sections
        with st.expander("📊 Categorical Features"):
            for feature in CATEGORICAL_FEATURES:
                st.write(f"• {feature}")
        
        with st.expander("🔢 Numerical Features"):
            for feature in NUMERICAL_FEATURES:
                st.write(f"• {feature}")
    
    with col2:
        st.markdown("**📥 Sample File:**")
        st.info("Download the sample CSV to see the correct format and column names.")
        
        # Create and offer sample CSV download
        sample_df = create_sample_csv()
        sample_csv = sample_df.to_csv(index=False)
        sample_b64 = base64.b64encode(sample_csv.encode()).decode()
        
        st.markdown(
            f'<a href="data:file/csv;base64,{sample_b64}" download="sample_customers.csv" '
            f'style="background-color: #4CAF50; color: white; padding: 10px 20px; '
            f'text-decoration: none; border-radius: 5px; display: inline-block;">📥 Download Sample CSV</a>',
            unsafe_allow_html=True
        )
        
        # Show sample data preview
        with st.expander("👀 Preview Sample Data"):
            st.dataframe(sample_df.head(3), use_container_width=True)
    
    st.markdown("---")
    
    # File upload
    st.subheader("📤 Upload Customer Data")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type="csv",
        help="Upload a CSV file with customer data for batch prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Validate data
            is_valid, errors, warnings = validate_csv_data(df)
            
            if errors:
                st.error("❌ **Data Validation Errors:**")
                for error in errors:
                    st.error(f"• {error}")
                return
            
            if warnings:
                st.warning("⚠️ **Data Validation Warnings:**")
                for warning in warnings:
                    st.warning(f"• {warning}")
            
            # Data cleaning options
            if warnings:
                st.subheader("🧹 Data Cleaning Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    handle_missing = st.selectbox(
                        "Handle Missing Values",
                        ["Remove rows", "Fill with median/mode", "Keep as is"],
                        help="Choose how to handle missing values"
                    )
                
                with col2:
                    if handle_missing != "Keep as is":
                        if st.button("🧹 Apply Cleaning"):
                            if handle_missing == "Remove rows":
                                df = df.dropna()
                                st.success(f"Removed rows with missing values. New shape: {df.shape}")
                            elif handle_missing == "Fill with median/mode":
                                # Fill numerical with median, categorical with mode
                                for col in NUMERICAL_FEATURES:
                                    if col in df.columns:
                                        df[col] = df[col].fillna(df[col].median())
                                for col in CATEGORICAL_FEATURES:
                                    if col in df.columns:
                                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                                st.success("Filled missing values with median/mode.")
            
            # Prediction section
            st.markdown("---")
            st.subheader("🔮 Generate Predictions")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    with show_loading_spinner(f"Processing {len(df)} customers with {selected_model_name}..."):
                        # Prepare data for prediction
                        required_features = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
                        prediction_data = df[required_features].copy()
                        
                        # Process predictions
                        results_df, error = process_batch_predictions(
                            prediction_data, selected_model, selected_model_name
                        )
                    
                    if error:
                        st.error(f"❌ Prediction failed: {error}")
                    else:
                        st.success("✅ Batch predictions completed successfully!")
                        
                        # Store results in session state
                        st.session_state['batch_results'] = results_df
                        st.session_state['batch_model'] = selected_model_name
            
            with col2:
                st.metric("Customers to Process", len(df))
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
    
    # Display results if available
    if 'batch_results' in st.session_state:
        results_df = st.session_state['batch_results']
        model_name = st.session_state['batch_model']
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Results summary
        summary = create_results_summary(results_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Customers",
                summary['total_customers']
            )
        
        with col2:
            st.metric(
                "Predicted Churn",
                summary['churn_count'],
                f"{summary['churn_rate']:.1f}%"
            )
        
        with col3:
            st.metric(
                "High Risk Customers",
                summary['high_risk'],
                help="Customers with >70% churn probability"
            )
        
        with col4:
            st.metric(
                "Potential Monthly Loss",
                f"${summary['potential_monthly_loss']:,.2f}",
                help="Monthly revenue at risk from predicted churners"
            )
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk category chart
            import plotly.graph_objects as go
            
            risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
            risk_values = [summary['low_risk'], summary['medium_risk'], summary['high_risk']]
            risk_colors = ['#28a745', '#ffc107', '#dc3545']
            
            fig = go.Figure(data=[go.Pie(
                labels=risk_labels,
                values=risk_values,
                marker_colors=risk_colors,
                hole=0.4
            )])
            
            fig.update_layout(
                title="Risk Category Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**📈 Business Impact Analysis:**")
            
            st.info(f"""
            **Financial Impact:**
            - Potential Annual Loss: ${summary['potential_annual_loss']:,.2f}
            - Average Monthly Charges: ${summary['avg_monthly_charges']:.2f}
            
            **Risk Distribution:**
            - 🟢 Low Risk: {summary['low_risk']} customers
            - 🟡 Medium Risk: {summary['medium_risk']} customers  
            - 🔴 High Risk: {summary['high_risk']} customers
            """)
            
            # Recommendations
            if summary['high_risk'] > 0:
                st.error(f"""
                **🚨 Immediate Actions Needed:**
                - {summary['high_risk']} customers need immediate attention
                - Prioritize retention campaigns
                - Review service quality issues
                """)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.multiselect(
                "Filter by Risk Level",
                options=['Low Risk', 'Medium Risk', 'High Risk'],
                default=['High Risk', 'Medium Risk'],
                help="Select risk levels to display"
            )
        
        with col2:
            churn_filter = st.multiselect(
                "Filter by Prediction",
                options=['Churn', 'No Churn'],
                default=['Churn'],
                help="Select prediction types to display"
            )
        
        with col3:
            show_columns = st.multiselect(
                "Select Columns to Display",
                options=list(results_df.columns),
                default=['customerID', 'Predicted_Churn', 'Probability_Percentage', 'Risk_Category', 'MonthlyCharges'],
                help="Choose which columns to show in the table"
            )
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if risk_filter:
            filtered_df = filtered_df[filtered_df['Risk_Category'].isin(risk_filter)]
        
        if churn_filter:
            filtered_df = filtered_df[filtered_df['Predicted_Churn'].isin(churn_filter)]
        
        if show_columns:
            filtered_df = filtered_df[show_columns]
        
        # Display filtered results
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )
        
        # Download options
        st.subheader("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Full results
            full_csv = results_df.to_csv(index=False)
            full_b64 = base64.b64encode(full_csv.encode()).decode()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            st.markdown(
                f'<a href="data:file/csv;base64,{full_b64}" download="churn_predictions_full_{timestamp}.csv" '
                f'style="background-color: #007bff; color: white; padding: 10px 20px; '
                f'text-decoration: none; border-radius: 5px; display: inline-block; margin: 5px;">📥 Full Results</a>',
                unsafe_allow_html=True
            )
        
        with col2:
            # High-risk customers only
            high_risk_df = results_df[results_df['Risk_Category'] == 'High Risk']
            if not high_risk_df.empty:
                high_risk_csv = high_risk_df.to_csv(index=False)
                high_risk_b64 = base64.b64encode(high_risk_csv.encode()).decode()
                
                st.markdown(
                    f'<a href="data:file/csv;base64,{high_risk_b64}" download="high_risk_customers_{timestamp}.csv" '
                    f'style="background-color: #dc3545; color: white; padding: 10px 20px; '
                    f'text-decoration: none; border-radius: 5px; display: inline-block; margin: 5px;">🚨 High Risk Only</a>',
                    unsafe_allow_html=True
                )
        
        with col3:
            # Summary report
            summary_report = pd.DataFrame([summary]).T
            summary_report.columns = ['Value']
            summary_csv = summary_report.to_csv()
            summary_b64 = base64.b64encode(summary_csv.encode()).decode()
            
            st.markdown(
                f'<a href="data:file/csv;base64,{summary_b64}" download="prediction_summary_{timestamp}.csv" '
                f'style="background-color: #28a745; color: white; padding: 10px 20px; '
                f'text-decoration: none; border-radius: 5px; display: inline-block; margin: 5px;">📊 Summary Report</a>',
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()