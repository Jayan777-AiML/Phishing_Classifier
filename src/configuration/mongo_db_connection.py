from pymongo import MongoClient

class MongoDBClient:
    client = None   #Singleton Pattern
    def __init__(self, uri: str):

        if MongoDBClient.client is None:
            MongoDBClient.client = MongoClient(uri)

        self.client = MongoDBClient.client