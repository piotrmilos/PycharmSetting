import matplotlib
import ml_utils
from ExperimentManagement.trainer import UrlTranslator
from mongo_resources2 import ExperimentStatus

matplotlib.use('Agg')

import argparse
import os
import shutil
import subprocess
import sys
from threading import Thread

from time import sleep
import signal
import traceback
from ExperimentManagement.config import MONGODB_HOST, MONGODB_PORT, DB_NAME, EXP_COLLECTION_NAME, WORKER_COLLECTION_NAME, \
    OLIMP_IP, STORAGE_PATH
from ExperimentManagement.db import create_mongodb_client
from ExperimentManagement.mongo_resources import Experiment, Worker

PING_SLEEP_SECONDS = 10

def concatenate_args_dict(args_dict):
    res = []
    for key, value in args_dict.iteritems():
        if value is not None:
            res.append('--' + str(key))
            res.append(str(value))
    return res


def create_parser():
    parser = argparse.ArgumentParser(description='TODO', fromfile_prefix_chars='@')
    parser.add_argument('--work-dir-path', type=str, help='TODO', required=True)
    parser.add_argument('--gpu-id', type=int, help='TODO', required=True)
    parser.add_argument('--log-path', type=str, help='TODO')
    parser.add_argument('--name', default='worker', type=str, help='TODO')
    parser.add_argument('--python-binary-path', default='python', type=str, help='TODO')
    parser.add_argument('--queues', nargs='+', default=None, type=str, help='TODO')
    parser.add_argument('--silent-child', action='store_true', help='TODO')

    return parser


def copy_code(url_translator, dump_dir_url, job_dir_path):
    print 'copy_code', dump_dir_url, job_dir_path
    resource, path = ml_utils.unpack(url_translator.split_url(dump_dir_url), 'resource', 'path')
    print 'res={}, path={}'.format(resource, path)
    if resource == 'storage':
        path = os.path.join(STORAGE_PATH, path[1:])
        scp_command = 'scp -oStrictHostKeyChecking=no -r ubuntu@{olimp_ip}:{remote_path} {local_path}'.format(
            olimp_ip=OLIMP_IP,
            remote_path=path,
            local_path=job_dir_path
        )
        print 'copying using scp command:', scp_command
        subprocess.call(scp_command, shell=True)
    else:
        dump_dir_path = UrlTranslator.url_to_path(dump_dir_url)
        shutil.copytree(src=dump_dir_path, dst=job_dir_path)


def fetch_exp(collection, queues):
    or_list = [{'queue': queue} for queue in queues]
    exp_fetched = collection.find_one_and_update({'status': ExperimentStatus.QUEUED.value,
                                                  '$or': or_list},
                                                 {'$set': {'status': ExperimentStatus.RUNNING.value}})
    return exp_fetched


class PingThread(Thread):
    def __init__(self, worker, sleep_seconds):
        super(PingThread, self).__init__()
        self.worker = worker
        self.sleep_seconds = sleep_seconds
        self.exit = 0
    def set_exit(self):
        self.exit = True

    def run(self):
        try:
            while not self.exit:
                print 'update ping'
                self.worker.update_ping()
                sleep(self.sleep_seconds)
        except:
            print traceback.format_exc()
            print 'Chujnia'


def handleTerm(sign, no):
    print "term", sign
    raise KeyboardInterrupt()

def test_tensorflow():
    import tensorflow as tf
    sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        ))

# TODO: worker env has to be predictible, etc, problems with scikit-image 0.11.3 vs 0.12.3,
# or make a binary ala bazel, with many dependencies packed
def main():
    # Please ALWAYS kill workers with SIGTERM. This will let it handle it gracefully.

    signal.signal(signal.SIGTERM, handleTerm)

    parser = create_parser()
    args = parser.parse_args()

    if args.log_path is not None:
        log_file = open(args.log_path, 'w')
        tee_stdout = ml_utils.Tee(log_file)
        tee_stderr = ml_utils.Tee(log_file)

        sys.stdout = tee_stdout
        sys.stderr = tee_stderr


    client = create_mongodb_client(MONGODB_HOST, MONGODB_PORT)
    db = client[DB_NAME]
    exp_collection = db[EXP_COLLECTION_NAME]
    worker_collection = db[WORKER_COLLECTION_NAME]

    worker = Worker(worker_collection)
    worker.set_status(Worker.Status.RUNNING)
    worker.set_log_url(args.log_path)
    worker.set_queues(args.queues)
    print 'PID', os.getpid()
    print 'log_path', args.log_path
    print 'name', args.name
    url_translator = UrlTranslator()

    try:
        worker.set_name(args.name)
        ping_thread = PingThread(worker, PING_SLEEP_SECONDS)
        ping_thread.start()

        print 'Worker id', str(worker.obj_id)

        work_dir_path = args.work_dir_path
        print 'work_dir_path', work_dir_path
        ml_utils.mkdir_p(work_dir_path)

        while True:

            exp_fetched = fetch_exp(exp_collection, args.queues)

            if exp_fetched is not None:
                worker.set_current_exp_id(str(exp_fetched['_id']))
                exp = Experiment(exp_collection, exp_id=str(exp_fetched['_id']))
                exp.set_last_worker_id(worker.get_worker_id())

                job_env = os.environ.copy()
                print list(exp_fetched.iterkeys())
                pythonpaths = exp_fetched['pythonpaths']
                pythonpath = ':'.join(pythonpaths)
                job_env.update({'PYTHONPATH': pythonpath, 'DEVICE': str(args.gpu_id)})
                job_id = ml_utils.id_generator(n=12)
                job_dir_path = os.path.join(work_dir_path, job_id)

                if 'dump_dir_url' not in exp_fetched:
                    print 'dum_dir_url should not be None'
                else:
                    copy_code(url_translator, exp_fetched['dump_dir_url'], job_dir_path)

                    # TODO: remove dependency on 'dl_dump'
                    job_cwd = job_dir_path

                    command = [args.python_binary_path, exp_fetched['argv'][0], '--exp-id', str(exp_fetched['_id'])]

                    print 20 * '='
                    print 'Executing:', command
                    print 'PYTHONPATH=', pythonpath
                    print 'job_dir_path', job_dir_path
                    print 'job_cwd', job_cwd
                    if args.silent_child:
                        # TODO: we should probably write 100 first lines of the output of the child, how to do it properly?
                        with open(os.devnull, 'w') as devnull:
                            child = subprocess.Popen(command, env=job_env, cwd=job_cwd, stdout=devnull, stderr=devnull)
                            child.wait()
                    else:
                        #test_tensorflow()
                        child = subprocess.Popen(command, env=job_env, cwd=job_cwd)
                        child.wait()

                print 'setting current_exp_id to None'
                worker.set_current_exp_id(None)
                print 'done!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            else:
                print 'sleeping'
                #sleep(5)
                sleep(60)

    except:
        ping_thread.set_exit()
        ping_thread.join()
        print 100 * 'Setting status exited'
        print traceback.format_exc()
        worker.set_status(Worker.Status.EXITED)
        print 'killing myself'
        tee_stdout.flush()
        tee_stderr.flush()
        raise
        #os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except:
        print 'unhandled exception'
        print traceback.format_exc()

        f = open('~/EXC_FROM_WORKER.txt', 'w')
        f.write(traceback.format_exc())
        f.close()


