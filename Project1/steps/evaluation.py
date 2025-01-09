import logging
import pandas as pd
from zenml import step

# The below function is used to evaluate the model

@step
def model_evaluate(df: pd.DataFrame) -> None:
    pass

