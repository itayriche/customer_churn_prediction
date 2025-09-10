"""Customer Churn Prediction Package

A machine learning package for predicting customer churn using multiple algorithms.
"""

__version__ = "0.1.0"
__author__ = "Customer Churn Prediction Team"

from . import config
from . import data_preprocessing
from . import model_training
from . import evaluation
from . import utils
from . import interpretability

__all__ = ["config", "data_preprocessing", "model_training", "evaluation", "utils", "interpretability"]