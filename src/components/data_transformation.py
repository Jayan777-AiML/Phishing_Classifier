import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from src.exception import CustomException
from src.logger import logging

from src.constant import *
from src.utils.main_utils import MainUtils



@dataclass
class DataTransformationConfig:
    data_transformation_dir = os.path.join(artifact_folder, 'data_transformation')
    transformed_train_file_path = os.path.join(data_transformation_dir, 'train.py') #We are converting into the npy format becoz it easy to export (like pushing into the s3 bucket and also light weight) 
    transformed_test_file_path = os.path.join(data_transformation_dir, 'test.npy')
    transformed_object_file_path = os.path.join(data_transformation_dir, 'preprocessing.pkl')

class DataTransfromation:
    def __init__(self, valid_data_dir):
        
        self.valid_data_dir = valid_data_dir 

        self.data_transformation_config = DataTransformationConfig()

        self.utils = MainUtils()

    @staticmethod
    def get_merged_batch_data(valid_data_dir: str) -> pd.DataFrame:
        
        #Method Name : get_merged_batch_data
        #Description : This method reads all the validated raw data from the valid_data_dir and return a pandas DataFrame containing the merged data.

        #Output : a pandas DataFrame containing the merged data
        #On Failure : Write an exception log and then raise an exception 

        #Version : 1.2
        #Revision : moved setup to cloud

        try:
            raw_files = os.listdir(valid_data_dir) #list down all the files in the valid_data_dir folder
            csv_data = []   #Saves all the necessary csv files in the list
            for filename in raw_files:
                data = pd.read_csv(os.path.join(valid_data_dir, filename))  #It's gonna join the directory path (address) with the file name to create the full path to the file.
                csv_data.append(data) 

            merged_data = pd.concat(csv_data)

            return merged_data
        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self):

        #Method Name: initiate_data_transformation
        #Description: This method initiate the data transformation component for the pipline

        #Output: data transformation artifact is created and returned
        #On Failure: write an exception log and then raise an exception

        #Version: 1.2
        #Revision: moved setup to cloud

        logging.info(
            "Entered initiate_data_transformation method of Data_TransFormation class"
        )

        try:
            dataframe = self.get_merged_batch_data(valid_data_dir= self.valid_data_dir)
            dataframe = self.utils.remove_unwanted_spaces(dataframe)
            dataframe.replace('?', np.NaN, inplace=True)

            X = dataframe.drop(columns = TARGET_COLUMN)
            y = np.where(dataframe[TARGET_COLUMN] == -1, 0, 1)

            sampler = RandomOverSampler()
            X_sampled, y_sampled = sampler.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2)

            preprocessor = SimpleImputer(strategy='most_frequent')


            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            preprocessor_path = (self.data_transformation_config.transformed_object_file_path)
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path = preprocessor_path,
                                   obj=preprocessor)
            
            return X_train_scaled, y_train, X_test_scaled, y_test, preprocessor_path
        
        except Exception as e:
            raise CustomException(e, sys) from e



