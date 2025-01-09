import logging

import pandas as pd
from zenml import step

class IngestData:
    '''
    Class to ingest data from the given path.

    '''

    def __init__(self, path: str):
        self.path = path

    def get_data(self):
        logging.info(f"Reading data from {self.path}")
        return pd.read_csv(self.path)


@step
def ingest_df(path: str) -> pd.DataFrame:

    '''
    Ingesting the data from the data_path .

    Args:
        data_path : path to the data
    Returns:
        pd.DataFrame : the injested data

    '''
    try:
        ingest_data = IngestData(path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error in Ingesting data: {e}")
        return None
    



















