import os
import sys
import pandas as pd

from dataclasses import dataclass

from datetime import datetime

from src.configuration.mongo_db_connection import MongoDBClient
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join("artifacts", "data_ingestion")


class DataIngestion:

    def __init__(
            self,
            uri,
            database_name,
            collection_name
    ):
        self.client = MongoDBClient(uri).client
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

        self.data_ingestion_config = DataIngestionConfig()

    def upload_csv(self, csv_path):
        try:
            logging.info(f"Reading CSV file: {csv_path}")

            df = pd.read_csv(csv_path)

            records = df.to_dict("records")

            logging.info(f"Inserting {len(records)} records into MongoDB")

            self.collection.insert_many(records)

            logging.info(
                f"{len(records)} records inserted into collection "
                f"{self.collection.name} of database {self.database.name}"
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def read_data(self):
        try:
            logging.info("Reading data from MongoDB collection")

            data = list(self.collection.find())

            df = pd.DataFrame(data)

            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            logging.info(f"Data loaded successfully. Shape: {df.shape}")

            return df

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self):
        """
        Method Name : initiate_data_ingestion

        Description :
            Reads data from MongoDB and saves it into the
            artifacts/data_ingestion directory as a CSV file.

        Output :
            Returns the raw data directory path.

        On Failure :
            Logs the exception and raises CustomException.
        """

        logging.info("Entered initiate_data_ingestion method of DataIngestion class")

        try:
            logging.info("Reading data from MongoDB")

            df = self.read_data()

            logging.info(
                f"Successfully read {df.shape[0]} rows and {df.shape[1]} columns."
            )

            raw_data_dir = self.data_ingestion_config.data_ingestion_dir

            logging.info(f"Creating directory: {raw_data_dir}")

            os.makedirs(raw_data_dir, exist_ok=True)

            ##raw_data_path = os.path.join(raw_data_dir, "phishing.csv") #It not changing the required name of file format

            date_stamp = datetime.now().strftime("%d%m%Y") # * digits, e.g. 09082026
            time_stamp = datetime.now().strftime("%H%M%S") # 6 digits, e.g. 1433210

            raw_data_path = os.path.join(
                raw_data_dir, f"phishing_{date_stamp}_{time_stamp}.csv"
            )

            logging.info(f"Saving raw data at: {raw_data_path}")

            df.to_csv(raw_data_path, index=False)

            logging.info("Raw data saved successfully.")

            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            return raw_data_dir

        except Exception as e:
            raise CustomException(e, sys) from e