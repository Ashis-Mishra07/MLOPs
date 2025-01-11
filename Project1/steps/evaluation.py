# The below function is used to evaluate the model

import logging
import pandas as pd
from zenml import step
from src.evaluation import Evaluation, MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated



import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker



@step(experiment_tracker = experiment_tracker.name)
def model_evaluate(model:RegressorMixin,
    X_test:pd.DataFrame,
    y_test:pd.DataFrame,                   
) -> Tuple[                                  # here wriiten that it will return score mean in which type
    Annotated[float , "r2_score"],
    Annotated[float , "rmse"],
    Annotated[float , "mse"]
]:
    
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        
        return mse, r2, rmse

    except Exception as e:
        logging.error(f'Error in evaluating model: {str(e)}')
        raise e
    

