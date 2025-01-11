import logging
import pandas as pd
from zenml import step

# The below function is used to train the model
from src.model_dev import Model, LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig


import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

# basically we have to log the models in the experiment tracker

@step(experiment_tracker = experiment_tracker.name)  # this will make it know that the model consistes of experiemnt tracker  
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig

) -> RegressorMixin:     # this is imported such that many other regression model can be imported at a time
    
    try:
        model = None
        if config.model_name == 'LinearRegression':
            mlflow.sklearn.autolog()  # this is used for autologging the model in the mlflow
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model

        else:
            logging.error('Model not found')
            raise Exception('Model not found')
    
    except Exception as e:
        logging.error(f'Error in training model: {str(e)}')
        raise e
    


