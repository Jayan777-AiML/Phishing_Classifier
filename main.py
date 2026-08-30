import os
from dotenv import load_dotenv

from src.components.data_ingestion import DataIngestion

load_dotenv()

URI = os.getenv("MONGODB_URI")

DATABASE_NAME = "db_sample"
COLLECTION_NAME = "sample_connection"

ingestion = DataIngestion(
    URI,
    DATABASE_NAME,
    COLLECTION_NAME
)

df = ingestion.read_data()

print(df.head())
print(df.shape)