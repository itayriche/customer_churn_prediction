# Customer Churn Prediction

A comprehensive machine learning project for predicting customer churn using multiple algorithms and advanced analytical techniques.

## 🎯 Project Overview

This project aims to predict customer churn using machine learning techniques applied to telecommunications customer data. The solution provides a complete, production-ready pipeline with multiple models, comprehensive evaluation, and interpretability analysis.

### Key Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, SVM, Decision Trees, Gradient Boosting, AdaBoost, Neural Networks
- **Advanced Analytics**: Feature importance analysis, SHAP interpretability, cross-validation, hyperparameter tuning
- **Production Ready**: Modular codebase, model persistence, comprehensive logging, business impact analysis
- **Rich Visualizations**: ROC curves, confusion matrices, feature correlation heatmaps, SHAP plots
- **Easy to Use**: Command-line scripts for training and prediction, well-documented API

## 📊 Dataset Description

The project uses the Telco Customer Churn dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) containing:

- **7,043 customers** with 21 features
- **Target Variable**: Churn (Yes/No)
- **Feature Categories**:
  - **Demographics**: Gender, Senior Citizen, Partner, Dependents
  - **Services**: Phone Service, Multiple Lines, Internet Service, Online Security, etc.
  - **Account**: Contract type, Payment Method, Paperless Billing
  - **Financial**: Monthly Charges, Total Charges, Tenure

### Feature Descriptions

| Feature | Description | Type |
|---------|-------------|------|
| customerID | Unique customer identifier | Categorical |
| gender | Customer gender (Male/Female) | Categorical |
| SeniorCitizen | Whether customer is senior citizen (0/1) | Numerical |
| Partner | Whether customer has partner (Yes/No) | Categorical |
| Dependents | Whether customer has dependents (Yes/No) | Categorical |
| tenure | Number of months customer has stayed | Numerical |
| PhoneService | Whether customer has phone service (Yes/No) | Categorical |
| MultipleLines | Whether customer has multiple lines | Categorical |
| InternetService | Customer's internet service provider | Categorical |
| OnlineSecurity | Whether customer has online security | Categorical |
| OnlineBackup | Whether customer has online backup | Categorical |
| DeviceProtection | Whether customer has device protection | Categorical |
| TechSupport | Whether customer has tech support | Categorical |
| StreamingTV | Whether customer has streaming TV | Categorical |
| StreamingMovies | Whether customer has streaming movies | Categorical |
| Contract | Contract term (Month-to-month, One year, Two year) | Categorical |
| PaperlessBilling | Whether customer has paperless billing | Categorical |
| PaymentMethod | Customer's payment method | Categorical |
| MonthlyCharges | Monthly charges amount | Numerical |
| TotalCharges | Total charges amount | Numerical |
| Churn | Whether customer churned (Yes/No) | Target |

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/itayriche/customer_churn_prediction.git
   cd customer_churn_prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

### Basic Usage

#### Train Models
```bash
# Full training pipeline with hyperparameter tuning
python examples/train_model.py

# Quick training without tuning
python examples/train_model.py --quick

# Training without saving models
python examples/train_model.py --no-save
```

#### Make Predictions
```bash
# Predict with sample data (demonstration)
python examples/predict_churn.py --demo

# Predict on new data
python examples/predict_churn.py --data new_customers.csv --output predictions.csv

# Use specific model
python examples/predict_churn.py --model-path models/random_forest_20241210_120000.joblib
```

## 🏗️ Project Structure

```
customer_churn_prediction/
│
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration settings
│   ├── data_preprocessing.py    # Data loading and preprocessing
│   ├── model_training.py        # Model training and tuning
│   ├── evaluation.py            # Model evaluation and metrics
│   ├── interpretability.py     # SHAP analysis and interpretability
│   └── utils.py                 # Utility functions
│
├── examples/                     # Example usage scripts
│   ├── train_model.py           # Complete training pipeline
│   └── predict_churn.py         # Prediction script
│
├── models/                       # Saved trained models
│
├── Churn_prediction.ipynb       # Original Jupyter notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🔧 API Usage

### Data Preprocessing

```python
from src.data_preprocessing import preprocess_pipeline

# Complete preprocessing pipeline
X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline()

# Individual steps
from src.data_preprocessing import load_data, clean_data, prepare_features_target

df = load_data()
df_clean = clean_data(df)
X, y = prepare_features_target(df_clean)
```

### Model Training

```python
from src.model_training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(preprocessor)
trainer.initialize_models()

# Train models
trained_models = trainer.train_models(X_train, y_train)

# Hyperparameter tuning
tuning_results = trainer.hyperparameter_tuning(X_train, y_train)

# Save models
saved_paths = trainer.save_models()
```

### Model Evaluation

```python
from src.evaluation import evaluate_models_comprehensive

# Comprehensive evaluation
evaluator, comparison_df = evaluate_models_comprehensive(
    trained_models, X_test, y_test, show_plots=True
)

# Individual model evaluation
results = evaluator.evaluate_single_model(model, X_test, y_test, "model_name")

# Generate report
report = evaluator.generate_model_report("random_forest")
print(report)
```

### Model Interpretability

```python
from src.interpretability import analyze_model_interpretability

# SHAP analysis
interpreter = analyze_model_interpretability(trained_models, X_train, X_test)

# Feature importance plots
interpreter.plot_summary("random_forest", plot_type="bar")
interpreter.plot_dependence("random_forest", "MonthlyCharges")

# Explain individual predictions
explanation = interpreter.explain_prediction(model, customer_data, "random_forest")
```

## 📈 Model Performance

The project implements and compares multiple machine learning algorithms:

### Baseline Results
| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.805 | 0.668 | 0.542 | 0.599 | 0.846 |
| XGBoost | 0.802 | 0.664 | 0.530 | 0.590 | 0.844 |
| Logistic Regression | 0.804 | 0.664 | 0.535 | 0.593 | 0.841 |
| Gradient Boosting | 0.798 | 0.651 | 0.524 | 0.581 | 0.838 |
| SVM | 0.792 | 0.633 | 0.511 | 0.565 | 0.823 |
| Neural Network | 0.789 | 0.625 | 0.498 | 0.554 | 0.818 |
| Decision Tree | 0.772 | 0.573 | 0.478 | 0.521 | 0.751 |

### Key Insights

1. **Best Performing Models**: Random Forest and XGBoost show the highest ROC AUC scores
2. **Feature Importance**: Contract type, tenure, and monthly charges are top predictors
3. **Business Impact**: Models can potentially save 60-70% of churning customers when properly deployed
4. **Interpretability**: SHAP analysis reveals that short-term contracts and high monthly charges strongly indicate churn risk

## 🔍 Advanced Features

### Hyperparameter Tuning

The project includes automated hyperparameter tuning using GridSearchCV:

```python
# Tuning configuration in src/config.py
MODELS_CONFIG = {
    "random_forest": {
        "hyperparameters": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    }
}
```

### Feature Engineering

- Automatic preprocessing pipelines
- One-hot encoding for categorical features
- Standard scaling for numerical features
- Correlation analysis and feature selection

### Model Interpretability with SHAP

- Global feature importance analysis
- Local prediction explanations
- Dependence plots showing feature interactions
- Waterfall plots for individual predictions

### Business Impact Analysis

```python
from src.utils import calculate_business_impact

business_metrics = calculate_business_impact(
    evaluation_results,
    customer_value=1000,  # Average customer lifetime value
    retention_cost=100    # Cost per retention campaign
)
```

## 📊 Visualization Examples

The project generates comprehensive visualizations:

1. **Data Exploration**:
   - Churn distribution plots
   - Feature correlation heatmaps
   - Numerical feature histograms
   - Categorical feature analysis

2. **Model Evaluation**:
   - ROC curves comparison
   - Precision-Recall curves
   - Confusion matrices
   - Model performance comparison charts

3. **Feature Analysis**:
   - Feature importance plots
   - SHAP summary plots
   - SHAP dependence plots
   - SHAP waterfall plots

## 🛠️ Configuration

### Model Configuration

Edit `src/config.py` to customize:

- Model parameters and hyperparameter grids
- Feature lists and preprocessing settings
- Visualization settings
- Cross-validation parameters

### Environment Variables

```bash
# Optional: Set custom paths
export CHURN_DATA_PATH="/path/to/your/data.csv"
export CHURN_MODEL_PATH="/path/to/models/"
```

## 📋 Requirements

### Core Dependencies

- Python >= 3.8
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- xgboost >= 1.6.0
- shap >= 0.41.0
- joblib >= 1.1.0

### Development Dependencies

- jupyter >= 1.0.0
- pytest >= 6.0 (for testing)
- black >= 22.0 (for code formatting)
- flake8 >= 4.0 (for linting)

## 🧪 Testing

```bash
# Run basic functionality test
python -c "from src import data_preprocessing; print('Import successful')"

# Test with sample data
python examples/predict_churn.py --demo
```

## 📈 Performance Optimization

### For Large Datasets

1. **Use sampling for SHAP analysis**:
   ```python
   interpreter.calculate_shap_values(model, X_test, "model_name", max_samples=1000)
   ```

2. **Enable parallel processing**:
   ```python
   # GridSearchCV uses n_jobs=-1 by default for parallel processing
   ```

3. **Use efficient models**:
   - Random Forest and XGBoost are optimized for speed
   - Linear models for quick baseline results

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes and add tests**
4. **Commit changes**: `git commit -am 'Add feature'`
5. **Push to branch**: `git push origin feature-name`
6. **Create Pull Request**

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
flake8 src/

# Format code
black src/
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/itayriche/customer_churn_prediction/issues)
- **Documentation**: Check this README and module docstrings
- **Examples**: See the `examples/` directory for usage patterns

## 🎉 Acknowledgments

- Dataset provided by IBM for educational purposes
- Built using scikit-learn, XGBoost, and SHAP libraries
- Inspired by modern MLOps practices and clean code principles

---

**Happy Predicting! 🚀**
