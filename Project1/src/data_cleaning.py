import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# create an abstract class for handling data

class DataStrategy(ABC):
    '''
    Abstract class defining strategy for handling data
    '''

    @abstractmethod
    def handle_data(self , data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:    # union means the output is combination of both the dataframe and series
        pass

class DataPreProcessStrategy(DataStrategy):
    '''
        Preprocess data
        '''
    def handle_data(self , data:pd.DataFrame) -> pd.DataFrame:
        
        try:
            data = data.drop(
                # The below cleaning isto completely drop the specific columns
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp"
                ],
                axis=1
            )
            # The below cleaning is to fill the missing values with mean / median values
            data['product_weight_g'].fillna(data['product_weight_g'].median(), inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(), inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(), inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(), inplace=True)
            data['review_comment_message'].fillna("No review",inplace=True)


            # The below cleaning is to choose the column having numeric values only
            data = data.select_dtypes(include=[np.number])

            # The below cleaning is to drop the columns which are not required
            cols_to_drop = ['customer_zip_code_prefix' , 'order_item_id']
            data = data.drop(cols_to_drop, axis=1)

            return data 

        except Exception as e:
            logging.error("Error in preprocessing data :{}".format(e))
            return e
        
    
# Now will implement data splitting strategy

class DataDivideStrategy(DataStrategy):
    '''
    Strategy for diving data into train and test
    '''

    def handle_data(self , data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:   
        try:
            X = data.drop('review_score', axis=1)   # review_score is the target variable
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error("Error in dividing data :{}".format(e))
            return e
        

# Here making a combine class for data dividing and cleaning
class DataCleaning:
    '''
    Class to handle data cleaning and dividing
    '''
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            self.strategy.handle_data(self.data)

        except Exception as e:
            logging.error("Error in handling data :{}".format(e))
            return e


'''


# Now how to run the program
if __name__ == '__main__':
    data = pd.read_csv('data/raw_data.csv')
    data_cleaning = DataCleaning(data, DataPreProcessStrategy())
    cleaned_data = data_cleaning.handle_data()

    
'''


