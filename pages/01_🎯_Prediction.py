"""
Interactive Customer Churn Prediction Interface

This page provides an interactive form for inputting customer data
and getting real-time churn predictions with explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from utils.web_utils import (
    load_models, load_custom_css, get_risk_category, 
    format_probability, create_probability_gauge,
    create_prediction_summary_card, validate_input_data,
    get_sample_customer_data, show_loading_spinner
)
from src.interpretability import ModelInterpreter

def create_input_form():
    """Create the customer data input form."""
    st.subheader("📝 Customer Information")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**👤 Demographics**")
        
        gender = st.selectbox(
            "Gender",
            options=['Female', 'Male'],
            help="Customer's gender"
        )
        
        senior_citizen = st.selectbox(
            "Senior Citizen",
            options=['No', 'Yes'],
            help="Whether customer is a senior citizen (65+)"
        )
        
        partner = st.selectbox(
            "Partner",
            options=['No', 'Yes'],
            help="Whether customer has a partner"
        )
        
        dependents = st.selectbox(
            "Dependents",
            options=['No', 'Yes'],
            help="Whether customer has dependents"
        )
        
        st.markdown("**📞 Phone & Internet Services**")
        
        phone_service = st.selectbox(
            "Phone Service",
            options=['No', 'Yes'],
            help="Whether customer has phone service"
        )
        
        multiple_lines = st.selectbox(
            "Multiple Lines",
            options=['No phone service', 'No', 'Yes'],
            help="Whether customer has multiple phone lines"
        )
        
        internet_service = st.selectbox(
            "Internet Service",
            options=['DSL', 'Fiber optic', 'No'],
            help="Type of internet service"
        )
        
        online_security = st.selectbox(
            "Online Security",
            options=['No', 'Yes', 'No internet service'],
            help="Whether customer has online security add-on"
        )
        
        online_backup = st.selectbox(
            "Online Backup",
            options=['No', 'Yes', 'No internet service'],
            help="Whether customer has online backup service"
        )
    
    with col2:
        st.markdown("**🛡️ Additional Services**")
        
        device_protection = st.selectbox(
            "Device Protection",
            options=['No', 'Yes', 'No internet service'],
            help="Whether customer has device protection plan"
        )
        
        tech_support = st.selectbox(
            "Tech Support",
            options=['No', 'Yes', 'No internet service'],
            help="Whether customer has tech support service"
        )
        
        streaming_tv = st.selectbox(
            "Streaming TV",
            options=['No', 'Yes', 'No internet service'],
            help="Whether customer has streaming TV service"
        )
        
        streaming_movies = st.selectbox(
            "Streaming Movies",
            options=['No', 'Yes', 'No internet service'],
            help="Whether customer has streaming movies service"
        )
        
        st.markdown("**💳 Account Information**")
        
        contract = st.selectbox(
            "Contract Type",
            options=['Month-to-month', 'One year', 'Two year'],
            help="Customer's contract type"
        )
        
        paperless_billing = st.selectbox(
            "Paperless Billing",
            options=['No', 'Yes'],
            help="Whether customer uses paperless billing"
        )
        
        payment_method = st.selectbox(
            "Payment Method",
            options=[
                'Electronic check', 
                'Mailed check', 
                'Bank transfer (automatic)',
                'Credit card (automatic)'
            ],
            help="Customer's payment method"
        )
    
    # Numerical features in a separate section
    st.markdown("**📊 Usage & Billing**")
    num_col1, num_col2, num_col3 = st.columns(3)
    
    with num_col1:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            help="Number of months customer has been with company"
        )
    
    with num_col2:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=65.0,
            step=0.01,
            help="Customer's monthly charges"
        )
    
    with num_col3:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=float(tenure * monthly_charges),
            step=0.01,
            help="Customer's total charges to date"
        )
    
    # Collect all inputs into a dictionary
    customer_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    return customer_data

def make_prediction(customer_data, model, model_name):
    """Make prediction and return results."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of churn (class 1)
        
        return {
            'prediction': prediction,
            'probability': probability,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        return {
            'prediction': None,
            'probability': None,
            'success': False,
            'error': str(e)
        }

def explain_prediction(customer_data, model, model_name):
    """Generate prediction explanation using SHAP."""
    try:
        # Initialize interpreter
        interpreter = ModelInterpreter()
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Get explanation
        explanation = interpreter.explain_prediction(model, df, model_name)
        
        return explanation
    
    except Exception as e:
        st.warning(f"Could not generate explanation: {str(e)}")
        return {}

def main():
    load_custom_css()
    
    st.title("🎯 Customer Churn Prediction")
    st.markdown("Enter customer information to predict churn probability and risk level.")
    
    # Model selection
    models = load_models()
    if not models:
        st.error("No trained models found. Please train models first.")
        return
    
    selected_model_name = st.selectbox(
        "Select Model for Prediction",
        options=list(models.keys()),
        help="Choose which machine learning model to use"
    )
    
    selected_model = models[selected_model_name]
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Use Sample Data", help="Fill form with sample customer data"):
            sample_data = get_sample_customer_data()
            st.session_state.update(sample_data)
            st.experimental_rerun()
    
    with col2:
        if st.button("🔄 Clear Form", help="Reset all fields"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.experimental_rerun()
    
    with col3:
        st.markdown("")  # Spacer
    
    # Create input form
    customer_data = create_input_form()
    
    # Validation and prediction
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
            # Validate input
            is_valid, errors = validate_input_data(customer_data)
            
            if not is_valid:
                st.error("Please fix the following errors:")
                for error in errors:
                    st.error(f"• {error}")
                return
            
            # Make prediction
            with show_loading_spinner("Making prediction..."):
                result = make_prediction(customer_data, selected_model, selected_model_name)
            
            if result['success']:
                # Store results in session state
                st.session_state['last_prediction'] = result
                st.session_state['last_customer_data'] = customer_data
                st.session_state['last_model_name'] = selected_model_name
                
                # Display results
                st.success("✅ Prediction completed successfully!")
                
            else:
                st.error(f"❌ Prediction failed: {result['error']}")
    
    with col2:
        if st.button("ℹ️ Model Info", use_container_width=True):
            with st.expander("Model Details", expanded=True):
                st.write(f"**Selected Model:** {selected_model_name}")
                st.write(f"**Model Type:** {type(selected_model).__name__}")
                if hasattr(selected_model, 'named_steps'):
                    st.write(f"**Pipeline Steps:** {list(selected_model.named_steps.keys())}")
    
    # Display prediction results if available
    if 'last_prediction' in st.session_state and st.session_state['last_prediction']['success']:
        result = st.session_state['last_prediction']
        customer_data = st.session_state['last_customer_data']
        model_name = st.session_state['last_model_name']
        
        prediction = result['prediction']
        probability = result['probability']
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Create summary card
        summary_html = create_prediction_summary_card(
            prediction, probability, model_name
        )
        st.markdown(summary_html, unsafe_allow_html=True)
        
        # Display gauge and details
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Probability gauge
            fig = create_probability_gauge(probability)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk assessment and recommendations
            category, color, emoji = get_risk_category(probability)
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {color};">
                <h4>{emoji} Risk Assessment: {category}</h4>
                <p><strong>Churn Probability:</strong> {format_probability(probability)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations based on risk level
            if category == "High Risk":
                st.warning("""
                **🚨 Immediate Action Recommended:**
                - Contact customer for retention offer
                - Review service satisfaction
                - Consider contract incentives
                - Prioritize customer support
                """)
            elif category == "Medium Risk":
                st.info("""
                **⚠️ Monitor Closely:**
                - Send satisfaction survey
                - Offer service upgrades
                - Check for service issues
                - Consider loyalty rewards
                """)
            else:
                st.success("""
                **✅ Customer Appears Stable:**
                - Continue regular service
                - Consider upselling opportunities
                - Maintain service quality
                - Periodic check-ins
                """)
        
        # Explanation section
        with st.expander("🔍 Prediction Explanation", expanded=False):
            explanation = explain_prediction(
                customer_data, 
                models[model_name], 
                model_name
            )
            
            if explanation:
                st.write("**Key factors influencing this prediction:**")
                
                if 'top_contributions' in explanation:
                    for contrib in explanation['top_contributions'][:5]:
                        feature = contrib['feature']
                        shap_value = contrib['shap_value']
                        feature_value = contrib['feature_value']
                        
                        impact = "increases" if shap_value > 0 else "decreases"
                        st.write(f"• **{feature}** (value: {feature_value:.3f}) {impact} churn likelihood")
                
                if 'explanation_text' in explanation:
                    st.write("**Summary:**")
                    st.write(explanation['explanation_text'])
            else:
                st.info("Prediction explanation not available for this model.")
        
        # Business impact section
        with st.expander("💰 Business Impact Analysis"):
            # Assume average customer value
            avg_monthly_value = customer_data['MonthlyCharges']
            annual_value = avg_monthly_value * 12
            
            if prediction == 1:  # Will churn
                st.error(f"""
                **Potential Revenue Loss:**
                - Annual Value: ${annual_value:,.2f}
                - Confidence: {format_probability(probability)}
                - Recommended Action: Immediate retention effort
                """)
            else:  # Will stay
                st.success(f"""
                **Expected Revenue Retention:**
                - Annual Value: ${annual_value:,.2f}
                - Confidence: {format_probability(1-probability)}
                - Recommended Action: Maintain service quality
                """)

if __name__ == "__main__":
    main()