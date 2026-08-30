import sys 
import os 
import logging
from typing import Dict, Tuple 

import numpy as np 
import pandas as pd 
import pickle 
import yaml 
import boto3 


from src.constant import *
from src.exception import CustomException
from src.logger import logging

class MainUtils:
    def __init__(self) -> None:
        pass

    def read_yaml_file(self, filename: str)-> dict:
        try: 
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)

        except Exception as e:
            raise CustomException(e, sys) from e 

    def read_schema_config_file(self)-> dict:
        try:
            schema_config = self.read_yaml_file(os.path.join("config", "schema.yaml"))

            return schema_config
        except Exception as e:
            raise CustomException(e, sys) from e 

# save_object() serializes and stores a Python object to disk using Pickle,
# while load_object() deserializes the saved file and restores the original object into memory.
    @staticmethod
    def save_object(file_path: str, obj: object) ->None:  #Becoz we are just logging and not returning anything
        logging.info("Entered the save_object method of MainUtils class")

        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

            logging.info("Exited the save_object method of MainUtils class")

        except Exception as e:
            raise CustomException(e, sys) from e 

    @staticmethod
    # preprocessor.pkl
    #         │
    #         ▼
    # open file
    #         │
    # pickle.load()
    #         │
    # returns Python object
    def load_object(file_path: str)-> object:
        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)         #trained preprocessor object (such as a ColumnTransformer)

            logging.info("Exited the load_object method of MainUtils class")

            return obj

        except Exception as e:
            raise CustomException(e, sys) from e 


    @staticmethod
    def upload_file(from_filename, to_filename, bucket_name):
        try:
            s3_resources = boto3.resource("s3")

            s3_resources.meta.client.upload_file(from_filename, bucket_name, to_filename)

        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def download_model(bucket_name, bucket_file_name, dest_file_name):
        try:
            s3_cliet = boto3.client("s3")

            s3_cliet.download_file(bucket_name, bucket_file_name, dest_file_name)

            return dest_file_name

        except Exception as e:
            raise CustomException(e, sys) from e     



    @staticmethod
    def remove_unwanted_spaces(data: pd.DataFrame) -> pd.DataFrame:
        try:
            df_without_space = data.apply(
                lambda x: x.str.strip() if x.dtype == 'object' else x)
            logging.info(
                "Unwanted spaces removal Successful. Exited the remove_unwanted_space method of the preprocessor class")
            return df_without_space
        except Exception as e:
            raise CustomException(e, sys)



    @staticmethod
    def identify_feature_types(dataframe: pd.DataFrame):
        data_types = dataframe.dtypes

        categorical_features = []
        continuous_features = []
        discrete_features = []

        for column, dtype in dict(data_types).items():
            unique_value = dataframe[column].nunique()

            if dtype == 'object' or unique_value < 10:
                categorical_features.append(column)
            elif dtype in [np.int64, np.float64]:
                if unique_value > 20:
                    continuous_features.append(column)
                else:
                    discrete_features.append(column)
            else:
                # Handle other data type if needed
                pass 
        return categorical_features, continuous_features, discrete_features

