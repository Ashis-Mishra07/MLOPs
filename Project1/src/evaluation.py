# same here also we will make abstarct classes and make other to import it 
import logging
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class Evaluation(ABC):
    '''
    Abstract class for evaluation
    '''

    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray): # here we have the true and predicted value and we can compare it
        pass



# creating strategy
class MSE(Evaluation):
    '''
    Evaluation Strategy by Mean Square Error
    '''

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info('Calculating Mean Square Error')
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE:{}".format(mse))
            return mse
        
        except Exception as e:
            logging.error(f'Error in calculating MSE: {str(e)}')
            raise e
        
class R2(Evaluation):
    '''
    Evaluation Strategy by R2
    '''

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info('Calculating R2')
            r2 = r2_score(y_true, y_pred)
            logging.info("R2:{}".format(r2))
            return r2
        
        except Exception as e:
            logging.error(f'Error in calculating R2: {str(e)}')
            raise e
        


class RMSE(Evaluation):
    '''
    Evaluation Strategy by Root Mean Square Error
    '''

    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info('Calculating Root Mean Square Error')
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("RMSE:{}".format(rmse))
            return rmse
        
        except Exception as e:
            logging.error(f'Error in calculating RMSE: {str(e)}')
            raise e
        

