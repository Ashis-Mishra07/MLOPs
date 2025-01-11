import logging 
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    '''
    Abstract class for model
    '''

    @abstractmethod
    def train(self , X_train , y_train):
        pass



class LinearRegressionModel(Model):
    '''
    Linear Regression Model
    '''

    def train(self , X_train , y_train , **kwargs):
        '''
        Train the model
        '''
        
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info('Model trained successfully')
            return reg
        except Exception as e:
            logging.error(f'Error in training model: {str(e)}')
            raise e
        



    
