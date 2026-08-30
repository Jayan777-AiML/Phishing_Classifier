import sys 
from typing import Generator, List, Tuple 
import os 
import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.naive_bayes import GaussianNB 
from sklearn.compose import ColumnTransformer 
from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV, train_test_split 
from src.constant import * 
from src.exception import CustomException
from src.logger import logging 
from src.utils.main_utils import MainUtils 

from dataclasses import dataclass 

@dataclass
class ModelTrainerConfig:
    model_trainer_dir = os.path.join(artifact_folder, 'model_trainer')
    trained_model_path = os.path.join(model_trainer_dir, 'trainer_model', 'model.pkl')
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', 'model.yaml')


class VisibilityModel:
    def __init__(self, preprocessing_object: ColumnTransformer, trained_model_object):

        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        logging.info("Entered predict method of srcTrainModel class")

        try:
            logging.info("Using the trained model to get the predictions")

            transfromed_feature = self.preprocessing_object.transform(X)

            logging.info("Used to trained model to get the predictions")

            return self.trained_model_object.predict(transfromed_feature)

        except Exception as e:
            raise CustomException(e, sys) from e

    def __repr__(self) -> str:
        """Return developer-friendly string representation showing the underlying model's class name."""
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self) -> str:
        """Return user-friendly string representation showing the underlying model's class name."""
        return f"{type(self.trained_model_object).__name__}()"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

        self.utils = MainUtils()

        self.models = {
            "GaussianNB": GaussianNB(),
            "XGBClassifier": XGBClassifier(objectives = 'binary:logistic'),
            "Logistic_Regression": LogisticRegression()
        }


    def evaluate_models(self, X_train, X_test, y_train, y_test, models):
        try:
            report = {} 
            print("Training Features: ", X_train.shape)

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred) 

                report[model_name] = test_model_score

            print("Evaluation Report: ", report) #prints the report
            return report

        except Exception as e:
            raise CustomException(e, sys)

        
    def finetune_best_model(self, best_model_object: object,
                            best_model_name,
                            X_train,
                            y_train) -> object:
#         Project/
#             │
#             └── model_selection/
#                 │
#                 └── model/
#                         │
#                         └── RandomForestClassifier/
#                                 │
#                                 └── search_param_grid

        try:    #The hyperparameter grid is NOT learned from the model. You write it yourself before training begins, ex:- max_depth, n_estimators, learning_rate.
            model_param_grid = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )["model_selection"]["model"][best_model_name]["search_param_grid"]


            grid_search = GridSearchCV(
                best_model_object, param_grid=model_param_grid, cv=5, n_jobs=-1, verbose=1)

            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            print("Best params are: ", best_params)

            finetuned_model = best_model_object.set_params(**best_params)   #The ** operator unpacks a dictionary into keyword arguments(named arguments).

            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys) 


    def initiate_model_trainer(self,
                               X_train,
                               y_train, 
                               X_test,
                               y_test,
                               preprocessor_path):

        try:
            logging.info(f"Splitting training and testing input and target features")

            logging.info("Loading preprocessor object")

            preprocessor = self.utils.load_object(file_path = preprocessor_path)

            logging.info(f"Extracting model config file path")

            model_report: dict = self.evaluate_models(
                X_train=X_train, 
                y_train=y_train,
                X_test=X_test,
                y_test=y_test, models=self.models
            )

            #To get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            #To get the best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model = self.models[best_model_name]

            best_model = self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_object=best_model,
                X_train=X_train,
                y_train=y_train 
            )


            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            best_model_score = accuracy_score(y_test, y_pred)

            print("Final Model Report: ", model_report, flush=True)
            print(f"best model name: {best_model_name} and score: {best_model_score}")

            if best_model_score < 0.5:
                raise Exception("No best model found with an accuracy greater than the threshold 0.6")

            logging.info(f"Best found model on both training and testing dataset")


            custom_model = VisibilityModel(
                preprocessing_object=preprocessor,
                trained_model_object=best_model
            )

            logging.info(f"Saving Model at path: {self.model_trainer_config.trained_model_path}")
            

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=custom_model,
            )

            self.utils.upload_file(
                from_filename = self.model_trainer_config.trained_model_path,
                to_filename = "model.pkl",
                bucket_name = AWS_S3_BUCKET_NAME
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)
        