import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig :
    preprocessor_obj_file_path= os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            #defining separate lists for numerical and categorical features of the dataset
            numerical_columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
       'Research',]
            categorical_columns =[]


            #building the pipeline
            logging.info("Pipeline Creation Started")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")), #takes care of missing values
                    ("scaler", StandardScaler()), #for normalizing data

                ]
            )

            #similarly we could've created a categorical pipeline, but since in this project we donot have any categorical variables I am skipping this step
            #cat_pipeline =[]

            logging.info("Pipeline Creation - COMPLETE")
            
            #to combine both the num and cat pipelines, we use the column transformer class
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline ,numerical_columns)
                    #we can add our cat_pipeline here
                ]
            )
            logging.info("Preprocessor Creation - COMPLETE")
            return preprocessor
            pass
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test datasets IMPORTED")
            logging.info("Obtaining Preprocessing object")
            preprocessing_obj = self.get_data_transformer_obj() #this object needs to be converted to a pickle file
            target_column_name = 'Chance of Admit '
            numerical_columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',
            'Research']

            

            # Dropping and isolating Y
            input_feature_train_df = train_df.drop(columns=[target_column_name], errors='ignore')
            target_feature_train_df = train_df[target_column_name]

            categorical_columns =[]
            #Dropping and Isolating the Y from df to get X for train
            input_feature_train_df = train_df.drop(target_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            #Dropping and Isolating the Y from df to get X for test
            input_feature_test_df = test_df.drop(target_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Starting Application of Preprocessing object")
            input_feature_train_arr  = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            #concatenating the X and Y as arrays for test and train separately
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            #now we will save our preprocessing_obj using the save_object function defined in the utils file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            logging.info("Saved Preprocessing Object")
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)
        
