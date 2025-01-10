import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreProcessStrategy, DataDivideStrategy

# so as to return the train test sample , python has annotated fromat to do it so
from typing import Tuple
from typing_extensions import Annotated

# The below function is used to clean the data

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    
    '''

    Cleans the data and divides it into train and test samples

    Args:
        df: The input data

    Returns:
        Tuple of X_train, X_test, y_train, y_test

    '''
    try:
        # first preprocessing
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        # second splitting / dividing the data
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()

        logging.info("Data cleaning and dividing is done successfully")

    except Exception as e:
        logging.error("Error in cleaning data :{}".format(e))
        return e
    
    


    



















