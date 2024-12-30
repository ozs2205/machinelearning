import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import dill
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train, Y_train,X_test,Y_test,models:dict, param:dict):
    try:
        report = {}
        for i in range(len(list(models))):
            para = param[list(models.keys())[i]]
            model = list(models.values())[i]
            gs = GridSearchCV(model,para, cv=4)
            gs.fit(X_train,Y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(Y_test, y_pred)
            report[list(models.keys())[i]]=r2
        
        return report
    except Exception as e:
        raise CustomException(e,sys)
