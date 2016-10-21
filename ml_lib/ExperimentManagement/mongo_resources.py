from time import sleep
import datetime

from bson import Binary, ObjectId
from enum import Enum
from pymongo.errors import AutoReconnect

from ExperimentManagement.db import create_mongodb_client



# NOTICE: taken from http://www.arngarden.com/2013/04/29/handling-mongodb-autoreconnect-exceptions-in-python-using-a-proxy/
from ExperimentManagement.mongo_resources2 import ExperimentStatus


def safe_mongocall(call):
    def _safe_mongocall(*args, **kwargs):
        for i in xrange(5):
            try:
                return call(*args, **kwargs)
            except AutoReconnect:
                sleep(pow(2, i))
        print 'Error: Failed operation!'

    return _safe_mongocall


class MongoResource(object):
    # WARNING!: Notice that wrapping in safe_mongocall could result in duplicate operations

    @safe_mongocall
    def push_to_array(self, arr_name, value):
        self.collection.update_one({'_id': self.obj_id},
                                   {'$push': {arr_name: value}})

    @safe_mongocall
    def set_field(self, field, value):
        self.collection.update_one({'_id': self.obj_id},
                                   {'$set': {field: value}})

    @safe_mongocall
    def get_field(self, field):
        res = self.collection.find_one({'_id': self.obj_id}, {field: 1})
        return res[field]


class Worker(MongoResource):
    class Status(Enum):
        RUNNING = 1
        EXITED = 2

    def __init__(self, collection):
        self.collection = collection
        new_worker = {'queues': []}
        self.obj_id = self.collection.insert_one(new_worker).inserted_id

    def set_name(self, name):
        self.set_field('name', name)

    # WARNING: time diff across machines!!
    def update_ping(self):
        self.set_field('last_ping', datetime.datetime.utcnow())

    def set_current_exp_id(self, exp_id):
        self.set_field('current_exp_id', exp_id)

    def set_status(self, status):
        self.set_field('status', status.value)

    def set_log_url(self, log_url):
        self.set_field('log_url', log_url)

    def set_queues(self, queues):
        print 'set_queues'
        for queue in queues:
            self.push_to_array('queues', queue)

    def get_worker_id(self):
        return str(self.obj_id)


# WARNING:
# We have to be really careful, sometimes we use name exp_id for str, but sometimes is ObjectId
class Experiment(MongoResource):

    def __init__(self, collection, owner=None, exp_id=None, tags=[]):
        self.collection = collection

        if exp_id is None:
            new_exp = {
                'owner': owner,
                'created': datetime.datetime.utcnow(),
                'last_ping': None,
                'epoch_data': [],
                'args_dict': {},
                'tags': tags
            }

            self.obj_id = self.collection.insert_one(new_exp).inserted_id
            self.set_status(ExperimentStatus.CREATED)
        else:
            self.obj_id = ObjectId(exp_id)

    ###### Getter, setters, updaters
    # TODO: automatize this, fix, or something else
    def get_exp_id(self):
        return str(self.obj_id)

    def set_queue(self, queue):
        self.set_field('queue', queue)

    def add_epoch_data(self, one_epoch_data):
        self.push_to_array('epoch_data', one_epoch_data)

    def add_tag(self, one_tag):
        self.push_to_array('tags', one_tag)

    def update_ping(self):
        self.set_field('last_ping', datetime.datetime.utcnow())

    def get(self):
        return self.collection.find_one({'_id': self.obj_id})

    def set_pythonpaths(self, pythonpaths):
        self.set_field('pythonpaths', pythonpaths)

    def set_dump_dir_url(self, dump_dir_url):
        self.set_field('dump_dir_url', dump_dir_url)

    def get_dump_dir_url(self):
        return self.get_field('dump_dir_url')

    def set_arch_url(self, arch_url):
        self.set_field('arch_url', arch_url)

    def set_valid_seed(self, valid_seed):
        self.set_field('valid_seed', valid_seed)

    def set_seed(self, seed):
        self.set_field('seed', seed)

    def set_args_dict(self, args_dict):
        self.set_field('args_dict', args_dict)

    def set_args_namespace(self, args):
        self.set_field('args_namespace', Binary(args, 128))

    def set_nof_params(self, nof_params):
        self.set_field('nof_params', nof_params)

    def set_weights_desc(self, weights_desc):
        self.set_field('weights_desc', weights_desc)

    def set_log_url(self, log_url):
        self.set_field('log_url', log_url)

    def set_name(self, name):
        self.set_field('name', name)

    def set_description(self, description):
        self.set_field('description', description)

    def set_exit_value(self, value):
        self.set_field('exit_value', value)

    def set_api_port(self, port):
        self.set_field('api_port', port)

    def set_api_host(self, host):
        self.set_field('api_host', host)

    def set_exit_traceback(self, exit_traceback):
        self.set_field('exit_traceback', exit_traceback)

    def set_status(self, status):
        self.set_field('status', status.value)

    def set_argv(self, argv):
        self.set_field('argv', argv)

    def set_prog_argv(self, prog_argv):
        self.set_field('prog_argv', prog_argv)

    def get_prog_argv(self):
        return self.get_field('prog_argv')

    def set_exp_dir_url(self, url):
        self.set_field('exp_dir_url', url)

    def set_pca_data_url(self, url):
        self.set_field('pca_data_url', url)

    def set_mean_data_url(self, url):
        self.set_field('mean_data_url', url)

    def set_last_worker_id(self, worker_id):
        self.set_field('last_worker_id', worker_id)

    @classmethod
    def find_all(cls, collection, owner=None, tag=None):
        if owner is not None:
            if tag is not None:
                exps = collection.find({'owner': owner, 'tags': tag})
            else:
                exps = collection.find({'owner': owner})
        else:
            exps = collection.find({})

        res = []
        for exp in exps:
            res.append(Experiment(collection, exp_id=exp['_id']))
        return res

if __name__ == '__main__':
    client = create_mongodb_client('localhost', 27017)
    db_name = 'exp_test'
    collection_name = 'experiments'

    db = client[db_name]
    collection = db[collection_name]

    updater = Experiment(collection, owner='maciek')
    updater.add_epoch_data({'lr': 3})
    updater.set_field('a.b.c.d', 3)
