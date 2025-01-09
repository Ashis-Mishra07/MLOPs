#NOTE:  ! means we are dealing with terminal commands


# ZenML is an open source library to build, run, and scale ML pipelines.


# installig the ZenML Library
'''

%pip install "zenml[server]"
!zenml integration install sklearn -y
%pip install pyparsing == 2.4.2  # required for colab

import IPython

#automatically restart kernel
IPython.Application.instance().kernel.do_shutdown(restart = True)



NGROK_TOPEN = "ercnrcnrcrncjkrc"    # this is required for colab

'''

# for Colab setup
'''

from zenml.environment import Environment

if Environment.in_google_colab():
    !pip install pyngrok
    !ngrok authtoken {NGROK_TOKEN}

'''

# Advantages of ZenML
'''
ML pipeline is more like a extension like sklearn where we typically do data acquisition ,
data preprocessing, model training, model evaluation and model deployment   before and after building model .

These pipelines helps in rerun of all the work and eliminating all the bugs and making the model easier 
to reproduce . All the data are keep on track and keep the record of the previous commit so that 
we can check the effectiveness of the  new versoning of the model.

We can also automate many operational tasks like retraining and redeploying models when the data changes 
or roling out new and improved models with CI / CD workflows .


'''

# install the zenml library and initialise it 
# !rm -rf .zen 
# !zenml init   # to initialise the zenml in the current directory


import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def train_test() -> None :
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size = 0.2, random_state = 42)
    model = SVC(gamma =0.01)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Model score: {score}")

train_test()

'''
Now turning the ML pipeline into ZenML pipeline

ML pipelines allows us to define our workflows in modular steps that we can then mix and match .

ZenML repository comes with  Importer , SCV trainer , Evaluator
It will import the data and train it and evaluate the score

'''

from zenml import step
from typing_extensions import Annotated
import pandas as pd
from typing import Tuple

@step
def importer() -> Tuple[                   # this function is used to return the training and testing parameter and doing the type checking 
    Annotated[np.ndarray,"X_train"] ,
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"]
]:
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test


@step
def svc_trainer(
    X_train: np.ndarray,           # this function will take the training data and return the model
    y_train: np.ndarray
) -> ClassifierMixin:              # the only work of the function is to return the classifier model
    model = SVC(gamma = 0.01)
    model.fit(X_train, y_train)
    return model


@step
def evaluator(
    model: ClassifierMixin,         # this function will take the model and return the score
    X_test: np.ndarray,
    y_test: np.ndarray
) -> float:                         # return the score in the floating format 
    score = model.score(X_test, y_test)
    return score

# Now its time to connect the functions using pipeline

from zenml import pipeline 

@pipeline
def digits_pipeline():
    X_train, X_test, y_train, y_test = importer()
    model = svc_trainer(X_train, y_train)
    score = evaluator(model, X_test, y_test)

digits_svc_pipeline = digits_pipeline()




