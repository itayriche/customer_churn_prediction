"""Configuration settings for customer churn prediction project."""

from typing import Dict, List, Any
import os

# Data configuration
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_SAVE_PATH = "models/"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Model configuration
MODELS_CONFIG: Dict[str, Any] = {
    "logistic_regression": {
        "class_name": "LogisticRegression",
        "params": {"random_state": RANDOM_STATE, "max_iter": 1000},
        "hyperparameters": {
            "C": [0.1, 1, 10, 100],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "random_forest": {
        "class_name": "RandomForestClassifier",
        "params": {"random_state": RANDOM_STATE, "n_estimators": 100},
        "hyperparameters": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        }
    },
    "svm": {
        "class_name": "SVC",
        "params": {"random_state": RANDOM_STATE, "probability": True},
        "hyperparameters": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }
    },
    "xgboost": {
        "class_name": "XGBClassifier",
        "params": {"random_state": RANDOM_STATE, "eval_metric": "logloss"},
        "hyperparameters": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "decision_tree": {
        "class_name": "DecisionTreeClassifier",
        "params": {"random_state": RANDOM_STATE},
        "hyperparameters": {
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    },
    "gradient_boosting": {
        "class_name": "GradientBoostingClassifier",
        "params": {"random_state": RANDOM_STATE},
        "hyperparameters": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    },
    "ada_boost": {
        "class_name": "AdaBoostClassifier",
        "params": {"random_state": RANDOM_STATE},
        "hyperparameters": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 1.0]
        }
    }
}

# Feature columns
CATEGORICAL_FEATURES: List[str] = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

NUMERICAL_FEATURES: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COLUMN: str = "Churn"
ID_COLUMN: str = "customerID"

# Visualization configuration
FIGURE_SIZE = (10, 8)
COLORS = ['cyan', 'orange']
PLOT_STYLE = 'seaborn-v0_8'

# Cross-validation configuration
CV_FOLDS = 5
SCORING_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Ensure model directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)