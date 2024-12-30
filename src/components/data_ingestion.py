import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path : str = os.path.join("artifacts", "test.csv")
    raw_data_path : str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        #when we create an instant of this DataIngestion class, it calls the DataIngestionConfig class and creates 3 files train,test,data

    def initiate_data_ingestion(self):
        #this block of code helps us read a dataset from a database
        logging.info("Entered the data ingestion method or component")
        try:
            #we can use any format like MongoDB client etc 
            df=pd.read_csv("D:\Machine Learning\\notebook\data_cleaned.csv")
            logging.info("Imported Dataset as df")

            #Using the following line of code, we make the required folders
            #We could've used either of train,test or raw path for creating the directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            #After the folder has been created we save the imported dataset to the raw path
            df = df.drop(['Unnamed: 0','Serial No.'], axis=1)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            #Starting Train-Test split and saving them to their respective files using their paths 
            logging.info("Train Test Split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of Data - COMPLETE")
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e,sys)

    
if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)

