import pathlib 
import sys, os 
# import dotenv import load_dotenv

import numpy as np 

from src.components.data_ingestion import DataIngestion 
from src.components.data_validation import DataValidation 
from src.components.data_transformation import DataTransfromation 
from src.components.model_trainer import ModelTrainer 
from src.exception import CustomException 

from src.constant import MONGO_DATABASE_NAME

# load_dotenv()

class TrainingPipeline:

    def start_data_ingestion(self):
        try:
            data_ingestion = DataIngestion(
                uri=os.getenv("MONGODB_URI"),
                database_name="db_sample",
                collection_name = "sample_connection"
            )
            raw_data_dir = data_ingestion.initiate_data_ingestion()
            return raw_data_dir

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, raw_data_dir):
        try:
            data_validation =  DataValidation(raw_data_store_dir=raw_data_dir)
            valid_data_dir = data_validation.initiate_data_validation()
            return valid_data_dir

        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transfromation(self, valid_data_dir):
        try:
            data_transformation = DataTransfromation(valid_data_dir=valid_data_dir)
            X_train, y_train, X_test, y_test, preporcessor_path = data_transformation.initiate_data_transformation()
            return X_train, y_train, X_test, y_test, preporcessor_path

        except Exception as e:
            raise CustomException(e, sys)


    def start_model_training(self, 
                             X_train: np.array,
                             y_train: np.array,
                             X_test: np.array,
                             y_test: np.array,
                             preprocessor_path: pathlib.Path):
        try:
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                X_train,
                y_train,
                X_test,
                y_test,
                preprocessor_path
            )

            return model_score, model_trainer

        except Exception as e:
            raise CustomException(e, sys) 


    def run_pipeline(self):
        try:
            raw_data_dir = self.start_data_ingestion()
            valid_data_dir = self.start_data_validation(raw_data_dir)
            X_train, y_train, X_test, y_test, preprocessor_path = self.start_data_transfromation(valid_data_dir)
            accuracy_score , model_trainer = self.start_model_training(X_train, y_train, X_test, y_test, preprocessor_path)

            print("Training completed. Trained model score: ", accuracy_score)
            return model_trainer

        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     train_pipeline = TrainingPipeline()
#     train_pipeline.run_pipeline()