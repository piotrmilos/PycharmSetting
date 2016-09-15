from pymongo import MongoClient

def create_mongodb_client(host, port):
    return MongoClient('mongodb://{host}:{port}/'.format(host=host, port=port))

