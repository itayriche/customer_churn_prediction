"""
Web utility functions for the Streamlit Customer Churn Prediction app.

This module provides helper functions for:
- Model loading and caching
- Data processing for web interface
- UI components and styling
- File operations
- Visualization helpers
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

"""
Web utility functions for the Streamlit Customer Churn Prediction app.

This module provides helper functions for:
- Model loading and caching
- Data processing for web interface
- UI components and styling
- File operations
- Visualization helpers
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

# Constants (copied from config to avoid import issues)
MODEL_SAVE_PATH = "models/"
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Simple data cleaning function (simplified from src)
def clean_data(df):
    """Clean the dataset."""
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing TotalCharges
    df = df.dropna(subset=['TotalCharges'])
    
    # Convert SeniorCitizen to string
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Drop customerID column if present
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    return df

@st.cache_data
def load_and_cache_data() -> pd.DataFrame:
    """Load and cache the dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
        df_cleaned = clean_data(df)
        return df_cleaned
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_resource
def load_models() -> Dict[str, Any]:
    """Load and cache all available trained models."""
    models = {}
    model_files = glob.glob(os.path.join(MODEL_SAVE_PATH, "*.joblib"))
    
    for model_file in model_files:
        try:
            model_name = Path(model_file).stem
            # Extract model type from filename (before timestamp)
            model_type = model_name.split('_')[0] + '_' + model_name.split('_')[1] if '_' in model_name else model_name
            models[model_type] = joblib.load(model_file)
        except Exception as e:
            st.warning(f"Could not load model {model_file}: {str(e)}")
    
    return models

def get_model_list() -> List[str]:
    """Get list of available model names."""
    try:
        models = load_models()
        return list(models.keys())
    except:
        return []

def set_page_config():
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/itayriche/customer_churn_prediction',
            'Report a bug': 'https://github.com/itayriche/customer_churn_prediction/issues',
            'About': "# Customer Churn Prediction App\nBuilt with Streamlit for machine learning model deployment."
        }
    )

def load_custom_css():
    """Load custom CSS styling."""
    css = """
    <style>
    /* Main styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    /* Risk level styling */
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #28a745;
    }
    
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #ffc107;
    }
    
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border-left: 4px solid #dc3545;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 0.25rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 0.25rem;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background-color: #d1ecf1;
        border-color: #bee5eb;
    }
    
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #007bff;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
    }
    
    /* Custom classes */
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def get_risk_category(probability: float) -> Tuple[str, str, str]:
    """
    Determine risk category based on churn probability.
    
    Args:
        probability: Churn probability (0-1)
        
    Returns:
        Tuple of (category, color, emoji)
    """
    if probability < 0.3:
        return "Low Risk", "#28a745", "🟢"
    elif probability < 0.7:
        return "Medium Risk", "#ffc107", "🟡"
    else:
        return "High Risk", "#dc3545", "🔴"

def format_probability(prob: float) -> str:
    """Format probability as percentage."""
    return f"{prob * 100:.1f}%"

def create_probability_gauge(probability: float) -> go.Figure:
    """Create a probability gauge chart."""
    category, color, _ = get_risk_category(probability)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def create_feature_importance_chart(feature_names: List[str], 
                                  importance_values: List[float], 
                                  title: str = "Feature Importance") -> go.Figure:
    """Create a horizontal bar chart for feature importance."""
    # Sort features by importance
    sorted_data = sorted(zip(feature_names, importance_values), key=lambda x: x[1], reverse=True)
    sorted_features, sorted_importance = zip(*sorted_data)
    
    fig = go.Figure(go.Bar(
        x=sorted_importance[:10],  # Top 10 features
        y=sorted_features[:10],
        orientation='h',
        marker_color='rgba(55, 128, 191, 0.7)',
        marker_line_color='rgba(55, 128, 191, 1.0)',
        marker_line_width=1
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig

def create_prediction_summary_card(prediction: int, 
                                 probability: float, 
                                 model_name: str) -> str:
    """Create a prediction summary card in HTML."""
    category, color, emoji = get_risk_category(probability)
    prediction_label = "Will Churn" if prediction == 1 else "Will Stay"
    
    card_html = f"""
    <div class="prediction-container">
        <h2>{emoji} Prediction Result</h2>
        <h3>{prediction_label}</h3>
        <p><strong>Churn Probability:</strong> {format_probability(probability)}</p>
        <p><strong>Risk Level:</strong> {category}</p>
        <p><strong>Model Used:</strong> {model_name.replace('_', ' ').title()}</p>
    </div>
    """
    
    return card_html

def validate_input_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input data for prediction.
    
    Args:
        data: Dictionary of input features
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check for required numerical features
    for feature in NUMERICAL_FEATURES:
        if feature not in data or data[feature] is None:
            errors.append(f"Missing required field: {feature}")
        elif not isinstance(data[feature], (int, float)):
            try:
                float(data[feature])
            except (ValueError, TypeError):
                errors.append(f"Invalid number format for {feature}")
    
    # Check for required categorical features
    for feature in CATEGORICAL_FEATURES:
        if feature not in data or data[feature] is None:
            errors.append(f"Missing required field: {feature}")
    
    # Business logic validation
    if 'tenure' in data and data['tenure'] is not None:
        if data['tenure'] < 0 or data['tenure'] > 100:
            errors.append("Tenure should be between 0 and 100 months")
    
    if 'MonthlyCharges' in data and data['MonthlyCharges'] is not None:
        if data['MonthlyCharges'] < 0 or data['MonthlyCharges'] > 200:
            errors.append("Monthly charges should be between $0 and $200")
    
    return len(errors) == 0, errors

def download_button(data: pd.DataFrame, filename: str, label: str = "Download CSV"):
    """Create a download button for DataFrame."""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def display_model_metrics(metrics: Dict[str, float]):
    """Display model performance metrics in a formatted way."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    
    with col4:
        st.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")

def create_confusion_matrix_heatmap(cm: np.ndarray, labels: List[str] = None) -> go.Figure:
    """Create a confusion matrix heatmap with correct orientation."""
    if labels is None:
        labels = ['No Churn', 'Churn']
    
    # Flip the confusion matrix vertically to correct orientation
    cm_flipped = np.flipud(cm)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_flipped,
        x=labels,
        y=labels[::-1],  # Reverse y-labels to match flipped matrix
        colorscale='Blues',
        text=cm_flipped,
        texttemplate="%{text}",
        textfont={"size": 20},
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=400,
        height=400,
        # Fix y-axis to prevent auto-reverse
        yaxis=dict(autorange=True)
    )
    
    return fig

def show_loading_spinner(message: str = "Processing..."):
    """Show a loading spinner with message."""
    return st.spinner(message)

def format_currency(amount: float) -> str:
    """Format amount as currency."""
    return f"${amount:,.2f}"

def get_sample_customer_data() -> Dict[str, Any]:
    """Get sample customer data for demonstration."""
    return {
        'gender': 'Female',
        'SeniorCitizen': 'No',
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 24,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.85,
        'TotalCharges': 1701.0
    }

# Simple interpretability class (simplified)
class SimpleInterpreter:
    """Enhanced model interpretation for web app."""
    
    def explain_prediction(self, model, customer_data, model_name, top_features=5):
        """Generate a comprehensive explanation of the prediction."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([customer_data])
            
            # Get prediction
            prediction = model.predict(df)[0]
            prediction_proba = model.predict_proba(df)[0]
            
            # Get feature importance based on model type
            feature_importance = self._get_feature_importance(model, model_name)
            
            # Get feature names from the preprocessor
            feature_names = self._get_feature_names(model)
            
            # Create feature-level explanations
            feature_explanations = []
            if feature_importance is not None and feature_names is not None:
                # Get top contributing features
                top_indices = np.argsort(np.abs(feature_importance))[-top_features:][::-1]
                
                for idx in top_indices:
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        importance = feature_importance[idx]
                        
                        # Get the actual feature value
                        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                            # Transform the data to get the actual feature value
                            try:
                                X_transformed = model.named_steps['preprocessor'].transform(df)
                                if hasattr(X_transformed, 'toarray'):
                                    X_transformed = X_transformed.toarray()
                                feature_value = X_transformed[0][idx] if idx < X_transformed.shape[1] else 0
                            except:
                                feature_value = 0
                        else:
                            feature_value = 0
                        
                        # Create human-readable explanation
                        explanation_text = self._create_feature_explanation(
                            feature_name, feature_value, importance, customer_data
                        )
                        
                        feature_explanations.append({
                            'feature': feature_name,
                            'importance': float(importance),
                            'value': feature_value,
                            'explanation': explanation_text
                        })
            
            # Risk assessment
            risk_level, risk_color = self._get_risk_assessment(prediction_proba[1])
            
            explanation = {
                'prediction': int(prediction),
                'prediction_label': 'Will Churn' if prediction == 1 else 'Will Stay',
                'churn_probability': float(prediction_proba[1]),
                'stay_probability': float(prediction_proba[0]),
                'confidence': float(np.max(prediction_proba)),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'feature_explanations': feature_explanations,
                'summary_text': self._create_summary_explanation(prediction, prediction_proba[1], model_name),
                'recommendations': self._generate_recommendations(prediction, prediction_proba[1], customer_data)
            }
            
            return explanation
            
        except Exception as e:
            st.error(f"Error explaining prediction: {e}")
            return {}
    
    def _get_feature_importance(self, model, model_name):
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                classifier = model.named_steps['classifier']
                if hasattr(classifier, 'feature_importances_'):
                    return classifier.feature_importances_
                elif hasattr(classifier, 'coef_'):
                    # For logistic regression, use absolute coefficients
                    return np.abs(classifier.coef_[0])
            return None
        except:
            return None
    
    def _get_feature_names(self, model):
        """Extract feature names from model pipeline."""
        try:
            if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
                preprocessor = model.named_steps['preprocessor']
                if hasattr(preprocessor, 'get_feature_names_out'):
                    return preprocessor.get_feature_names_out()
                elif hasattr(preprocessor, 'transformers_'):
                    # Handle ColumnTransformer
                    feature_names = []
                    for name, transformer, columns in preprocessor.transformers_:
                        if name != 'remainder':
                            if hasattr(transformer, 'get_feature_names_out'):
                                names = transformer.get_feature_names_out(columns)
                                feature_names.extend(names)
                            else:
                                feature_names.extend(columns)
                    return feature_names
            return None
        except:
            return None
    
    def _create_feature_explanation(self, feature_name, feature_value, importance, customer_data):
        """Create human-readable explanation for a feature."""
        impact = "increases" if importance > 0 else "decreases"
        strength = "strongly" if abs(importance) > 0.1 else "moderately"
        
        # Map technical feature names to readable names
        readable_name = feature_name.replace('_', ' ').title()
        
        return f"{readable_name} {strength} {impact} churn probability"
    
    def _get_risk_assessment(self, churn_prob):
        """Determine risk level and color."""
        if churn_prob >= 0.7:
            return "High Risk", "red"
        elif churn_prob >= 0.4:
            return "Medium Risk", "orange"
        else:
            return "Low Risk", "green"
    
    def _create_summary_explanation(self, prediction, churn_prob, model_name):
        """Create summary explanation text."""
        if prediction == 1:
            return f"The {model_name.replace('_', ' ').title()} model predicts this customer is likely to churn with {churn_prob*100:.1f}% probability."
        else:
            return f"The {model_name.replace('_', ' ').title()} model predicts this customer will likely stay with {(1-churn_prob)*100:.1f}% probability."
    
    def _generate_recommendations(self, prediction, churn_prob, customer_data):
        """Generate actionable recommendations."""
        recommendations = []
        
        if churn_prob > 0.5:  # High churn risk
            if customer_data.get('Contract') == 'Month-to-month':
                recommendations.append("💡 Offer annual contract with discount")
            
            if customer_data.get('TechSupport') == 'No':
                recommendations.append("🛠️ Provide complimentary tech support")
            
            if customer_data.get('MonthlyCharges', 0) > 70:
                recommendations.append("💰 Consider loyalty discount or price adjustment")
            
            recommendations.append("📞 Schedule retention call within 7 days")
        else:  # Low churn risk
            recommendations.append("✅ Customer appears satisfied - monitor quarterly")
            recommendations.append("📈 Consider upselling additional services")
        
        return recommendations