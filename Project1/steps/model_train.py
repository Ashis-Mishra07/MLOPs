import logging
import pandas as pd
from zenml import step

# The below function is used to train the model

@step
def train_model(df: pd.DataFrame) -> None:
    pass

