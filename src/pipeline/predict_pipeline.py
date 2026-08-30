import shutil
import os, sys 
import pandas as pd
from src.logger import logging 

from src.exception import CustomException 
from flask import request 
from src.constant import *  
from src.utils.main_utils import MainUtils

from dataclasses import dataclass 

@dataclass 
class PredictionFileDetail: #(Declare member variables)
    prediction_output_dirname: str = "predictions" #All the prediction(folder) output will get save here
    prediction_file_name: str = "predicted_file.csv" #All the predictions from the model will get saved in csv file in Predicion(folder)
    prediction_file_path: str = os.path.join(prediction_output_dirname, prediction_file_name)

 
class PredictionPipeline:
    def __init__(self, request):
        self.request = request
        self.utils = MainUtils()
        self.prediction_file_detail = PredictionFileDetail()


    def save_input_files(self)->str:
        '''
        Method Name: save_input_file
        Description: This method saves the input files to the prediction artifacts directoty
        
        Output: Input DataFrame
        On Failure: Write an exception log and then raise an exception
        
        Version: 1.2
        Revision: moved setup to cloud
        '''

        try:
            pred_file_input_dir = "prediction_artifacts" #Saves the user uploaded file
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file'] #Reads the user uploded file (user passes the file)
            pred_file_path = os.path.join(pred_file_input_dir, input_csv_file.filename)

            input_csv_file.save(pred_file_path) #Saves the input file

            return pred_file_path

        except Exception as e:
            raise CustomException(e, sys)


    def predict(self, features):
        '''
        Method Name: predict
        Description: loading the already-trained ML model and using it to make predictions on the input data
        
        '''

        try:
            model_path = self.utils.download_model(
                bucket_name=AWS_S3_BUCKET_NAME,
                bucket_file_name = "model.pkl",
                dest_file_name = "model.pkl"
            )

            model = self.utils.load_object(file_path = model_path)   #In python whenever we store a file in a pkl format we called it as object (dest_file_name).
                                                                    #We are going to get the trained model from aws s3 bucket,
                                                                    #From aws bucket we are going to download the model.pkl file and then we are going to load it using load_object method of MainUtils class and saving it in model.pkl directory,
                                                                    # then we are going to use that model (preds) and predict.

            pred = model.predict(features)

            return pred
        
        except Exception as e:
            raise CustomException(e, sys)


    def get_predicted_dataframe(self, input_dataframe_path: pd.DataFrame):
        '''
        Method Name: get_predicted_dataframe
        Description: This method takes the input dataframe and returns the predicted dataframe with the new predictions column
        
        Output: Predicted DataFrame
        On Failure: Write an exception log and then raise an exception
        
        Version: 1.2
        Revision: moved setup to cloud
        '''

        try:
            prediction_column_name : str = TARGET_COLUMN
            input_dataframe = pd.read_csv(input_dataframe_path)

            print("Prediction Features: ", input_dataframe.columns.tolist())
            input_dataframe = input_dataframe.drop(columns=["Result"], errors = "ignore")

            prediction = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in prediction]
            target_column_mapping = {0: 'phishing', 1: 'safe'}

            

            input_dataframe['Mapped Result'] = input_dataframe[prediction_column_name].map(target_column_mapping)

            os.makedirs(self.prediction_file_detail.prediction_output_dirname, exist_ok=True)
            input_dataframe.to_csv(self.prediction_file_detail.prediction_file_path, index=False)
            logging.info("Prediction Completed.")

        except Exception as e:
            raise CustomException(e, sys) from e



    def run_pipline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_file_detail 

        except Exception as e:
            raise CustomException(e, sys)
