import copy
import shutil
import subprocess
from mock import Mock, MagicMock
from threading import Thread
from time import sleep
import re

import matplotlib

from ml_utils import timestamp_alt_str, mkdir_p
import ml_utils
from ExperimentManagement.mongo_resources2 import ExperimentStatus
import training_utils

matplotlib.use('Agg')

import argparse
import random
import signal
import traceback

from bokeh import embed
from bokeh.io import push, cursession, output_server
from bokeh.models import HoverTool
from bokeh.plotting import figure
from bunch import Bunch
from ExperimentManagement.config import MONGODB_HOST, MONGODB_PORT, BOKEH_HOST, BOKEH_PORT, PUBLIC_BOKEH_PORT, \
    PUBLIC_BOKEH_HOST
from ExperimentManagement.db import create_mongodb_client
from ExperimentManagement.mongo_resources import Experiment
from ml_utils import TimeSeries, BokehTimeseriesObserver, id_generator
import os
import sys

DL_CODE_DUMP_SUBDIR = 'dl_dump'

class ExitHandlerThread(Thread):
    def __init__(self, command_receiver):
        super(ExitHandlerThread, self).__init__()
        self.sleep_seconds = 2
        self.do_exit = 0
        self.command_receiver = command_receiver

    def do_stop(self):
        self.do_exit = 1

    def run(self):
        while not self.do_exit:
            if self.command_receiver.do_stop:
                os.kill(os.getpid(), signal.SIGTERM)
            sleep(self.sleep_seconds)


class UrlTranslator(object):
    # (resource_name, env_var_name)
    r = [
        ('storage', 'STORAGE'),
    ]

    def get_url_to_path_rules(self):
        res = []
        for (a, b) in self.r:
            pattern = '^//' + a + '/'
            subl = os.environ[b] + '/'
            res.append((pattern, subl))
        return res

    def get_path_to_url_rules(self):
        res = []
        for (a, b) in self.r:
            pattern = '^' + os.environ[b] + '/'
            subl = '//' + a + '/'
            res.append((pattern, subl))
        return res

    def apply_rules(self, rules, w):
        w = copy.copy(w)
        if w is None:
            return w

        for pattern, subl in rules:
            w = re.sub(pattern, subl, w)
        return w

    def split_url(self, url):
        regex = r"""//(\w*)(/.*)"""
        m = re.compile(regex).match(url)
        return Bunch(resource=m.groups(0)[0], path=m.groups(0)[1])

    def url_to_path(self, url):
        res = self.apply_rules(self.get_url_to_path_rules(), url)
        return res

    def path_to_url(self, path):
        res = self.apply_rules(self.get_path_to_url_rules(), path)
        return res


class Trainer(object):
    def __init__(self):
        pass

    def get_url_translator(self):
        return UrlTranslator()

    def transform_urls_to_paths(self, args):
        url_translator = self.get_url_translator()
        regex = re.compile('.*_url$')
        keys = copy.copy(vars(args))
        for arg in keys:
            if regex.match(arg):
                new_arg = re.sub('_url$', '_path', arg)
                setattr(args, new_arg, url_translator.url_to_path(getattr(args, arg)))
        return args

    def _create_timeseries_and_figures(self, channels, figures_schema, plot_height=500, plot_width=500):
        ts = Bunch()
        for ts_name in channels:
            ts.__setattr__(ts_name, TimeSeries())

        # NOTICE: colors are taken from blocks project
        # http://www.w3schools.com/html/html_colornames.asp
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#000000', '#0000FF', '#7FFF00', '#006400', '#2F4F4F']

        for figure_title, l in figures_schema.iteritems():
            hover = HoverTool(
                tooltips=[
                    ("(x,y)", "($x, $y)"),
                ]
            )
            tools = ['pan', 'wheel_zoom', 'box_zoom', 'reset', 'resize'] + [hover]
            p = figure(title=figure_title, plot_width=plot_width, plot_height=plot_height, tools=tools)

            for idx, (ts_name, line_name, mean_freq) in enumerate(l):
                line = p.line([], [], name=line_name, legend=line_name, line_color=colors[idx], line_width=1)
                renderer = p.select(dict(name=line_name))
                ds = renderer[0].data_source
                observer = BokehTimeseriesObserver(datasource=ds, add_freq=mean_freq)
                getattr(ts, ts_name).add_add_observer(observer)

            self.exp.push_to_array('bokeh_tags', embed.autoload_server(p, cursession()))

        push()
        return ts

    def create_bokeh_session(self):
        url = 'http://{host}:{port}/'.format(host=PUBLIC_BOKEH_HOST, port=PUBLIC_BOKEH_PORT)
        output_server(docname=str(self.exp.obj_id), url=url)
        print 'Done'

    @classmethod
    def fetch_or_create_experiment(self, owner=None, exp_id=None, tags=None):
        if tags is None: # This makes sense, really !!!
            tags = []

        client = create_mongodb_client(MONGODB_HOST, MONGODB_PORT)
        db_name = 'zeus'
        collection_name = 'experiments'
        db = client[db_name]
        collection = db[collection_name]
        exp = Experiment(collection=collection, owner=owner, exp_id=exp_id, tags=tags)
        return exp

    def save_model(self, model, file_name):
        print 'ModelPath', file_name
        model_path = self.saver.save_train_state_new(model, file_name)

        return model_path

    def init_command_receiver(self, ssh_reverse_host=None):
        # WARNING!!!!: currently when we kill a process at the end, ssh is killed but it seems the port is
        # still open at the remote machine.

        # INFO: make sure that on AWS this port range is reachable
        # WARNING: possibility of a collision
        api_port = random.randint(8000, 13000)

        if (ssh_reverse_host is not None) or 'MY_PUBLIC_IP' not in os.environ:
            api_host = ssh_reverse_host
            if api_host is None:
                api_host = BOKEH_HOST

            # IMPORTANT: You have to set GatewayPorts option in sshd config on remote host,
            # for this command to work properly, see http://superuser.com/questions/588591/how-to-make-ssh-tunnel-open-to-public
            # for more info
            command = 'autossh -M {monitor_port} -R 0.0.0.0:{remote_port}:localhost:{local_port} ubuntu@{remote_host} -N'.format(
                monitor_port=api_port + 1,
                remote_port=api_port,
                                                                                                local_port=api_port,
                                                                                                remote_host=api_host)
            print command
            r = subprocess.Popen(command, shell=True)
        else:
            api_host = os.environ['MY_PUBLIC_IP']

        self.exp.set_api_port(api_port)
        self.exp.set_api_host(api_host)
        self.command_receiver = training_utils.run_command_receiver(api_port)
        print 'command receiver ', type(self.command_receiver), dir(self.command_receiver)

    def start_exit_handler_thread(self, command_receiver):
        print 'Staring exit handler!'
        self.exit_handler_thread = ExitHandlerThread(command_receiver)
        self.exit_handler_thread.start()

    def stop_exit_handler_thread(self):
        print 'Stopping exit handler!'
        self.exit_handler_thread.do_stop()
        self.exit_handler_thread.join()
        print 'Exit handler stopped!'

    def create_control_parser(self, default_owner):
        parser = argparse.ArgumentParser(description='TODO', fromfile_prefix_chars='@')
        parser.add_argument('--tags', type=str, help='TODO', nargs='+', default=None)
        parser.add_argument('--queue', type=str, default=None, help='TODO')
        parser.add_argument('--only-create', action='store_true', help='TODO')
        parser.add_argument('--no-exp', action='store_true', help='TODO')
        parser.add_argument('--exp-id', type=str, default=None, help='TODO')
        parser.add_argument('--copy-dump', type=str, default=None, help='TODO')
        parser.add_argument('--no-dump', action='store_true', help='TODO')
        parser.add_argument('--owner', type=str, default=default_owner, help='TODO')
        parser.add_argument('--exp-dir-url', type=str, default=None, help='TODO')
        parser.add_argument('--exp-parent-dir-url', type=str, default=None, help='TODO')

        return parser

    def main(self, default_owner, paths_to_dump, pythonpaths=['/', '/ml_lib'], default_tags=[]):
        # NOTICE:
        # not sure of all the intricacies of processes, but it seems to work, we create new process group,
        # every subprocess will belong to this group, and then at the end we send signal to the group(SIGKILL)
        os.setpgrp()

        parser = self.create_parser()
        control_parser = self.create_control_parser(default_owner=default_owner)

        control_args, prog_argv = control_parser.parse_known_args(sys.argv[1:])
        print control_args
        print prog_argv
        control_args = self.transform_urls_to_paths(control_args)

        if control_args.exp_id is not None:
            # We should fetch args from db, but we will do it just before running actual main
            exp = self.fetch_or_create_experiment(exp_id=control_args.exp_id)
            exp_fetched = exp.get()
            exp_dir_path = self.get_url_translator().url_to_path(exp_fetched['exp_dir_url'])
        else:
            owner = control_args.owner

            tags = default_tags if control_args.tags is None else control_args.tags

            if control_args.no_exp:
                exp = MagicMock()
                rand_id = str(id_generator(10))
                exp.get_exp_id = MagicMock(return_value=rand_id)
            else:
                exp = self.fetch_or_create_experiment(exp_id=control_args.exp_id, owner=owner, tags=tags)

            # WARNING: hacks!!!, to make this runnable by worker
            only_create = control_args.only_create
            queue = control_args.queue
            print 'SAViNG PROG argv!!!!!!!!'
            exp.set_argv([sys.argv[0]] + prog_argv)
            exp.set_prog_argv(prog_argv)

            exp_dir_path = None
            print vars(control_args)
            if control_args.exp_dir_path:
                exp_dir_path = control_args.exp_dir_path

            elif control_args.exp_parent_dir_path:
                exp_dir_path = os.path.join(control_args.exp_parent_dir_path, '{exp_id}_{random_id}_{timestamp}'.format(
                    exp_id=exp.get_exp_id(),
                    random_id=id_generator(5),
                    timestamp=timestamp_alt_str()
                    )
                )
            else:
                raise RuntimeError('exp_dir_path is not present!!!')


            exp.set_exp_dir_url(self.get_url_translator().path_to_url(exp_dir_path))
            exp.set_pythonpaths(pythonpaths)

            if paths_to_dump is not None and exp_dir_path is not None and not control_args.no_dump and control_args.copy_dump is None:
                print 'Dumping paths:', paths_to_dump
                try:

                    dump_dir_path = os.path.join(exp_dir_path, DL_CODE_DUMP_SUBDIR)
                    print 'dump_dir', dump_dir_path
                    mkdir_p(dump_dir_path)
                    for path_to_dump in paths_to_dump:
                        if ml_utils.DEEPLEARNING_HOME is None:
                            raise RuntimeError('ml_utils.DEEPLEARNING_HOME should not be None')

                        src = os.path.join(ml_utils.DEEPLEARNING_HOME, path_to_dump)
                        dst = os.path.join(dump_dir_path, path_to_dump)
                        dst_parent_dir = os.path.split(dst)[0]
                        mkdir_p(dst_parent_dir)
                        os.system('cp -r {src} {dst}'.format(src=src, dst=dst))
                        # shutil.copytree(src=os.path.join(ml_utils.DEEPLEARNING_HOME, path_to_dump),
                        #                 dst=os.path.join(dump_dir_path, path_to_dump))
                except OSError as e:
                    print e
                    raise
                exp.set_dump_dir_url(self.get_url_translator().path_to_url(dump_dir_path))
            elif control_args.copy_dump is not None:
                other_exp_id = control_args.copy_dump
                print 'Using dl_dump from ', other_exp_id
                other_exp = self.fetch_or_create_experiment(exp_id=other_exp_id)
                other_dump_dir_url = other_exp.get_dump_dir_url()
                other_dump_dir_path = self.get_url_translator().url_to_path(other_dump_dir_url)
                try:
                    my_dump_dir_path = os.path.join(exp_dir_path, DL_CODE_DUMP_SUBDIR)
                    shutil.copytree(src=other_dump_dir_path, dst=my_dump_dir_path)
                except OSError as e:
                    print e
                    raise
                exp.set_dump_dir_url(self.get_url_translator().path_to_url(my_dump_dir_path))
            else:
                print 'NO CODE DUMPING!!!, are you sure?'

            exp.set_exit_value(None)

            if only_create:
                print 'Only create, exiting!'
                return 0

            if queue:
                exp.set_status(ExperimentStatus.QUEUED)
                exp.set_queue(queue)
                print 'Only queue, exiting!'
                return 0
        try:
            # The actual arguments that are passed to the program are modified, by the following lines.
            if not control_args.no_exp:
                prog_argv = exp.get_prog_argv()

            prog_args = parser.parse_args(prog_argv)
            prog_args = self.transform_urls_to_paths(prog_args)
            self.install_sigterm_handler()
            res = self.go(exp, prog_args, exp_dir_path)

            exp.set_exit_value(res)

        except:
            print traceback.format_exc()
            exp.set_exit_traceback(traceback.format_exc())
            exp.set_exit_value(-1)

        exp.set_status(ExperimentStatus.COMPLETED)
        print 'Sending SIGKILL to the process group :( bye, bye'
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


    # The user have to define go function
    def go(self, exp, args, exp_dir_path):
        raise NotImplementedError()

    def create_timeseries_and_figures(self):
        raise NotImplementedError()

    @classmethod
    def create_parser(cls):
        parser = argparse.ArgumentParser(description='TODO', fromfile_prefix_chars='@')
        return parser