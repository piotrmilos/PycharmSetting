from ExperimentManagement.db import create_mongodb_client

__author__ = 'maciek'

client = create_mongodb_client('localhost', 27017)
db_name = 'exp_test'
collection_name = 'experiments'
db = client[db_name]
collection = db[collection_name]


collection.remove()
