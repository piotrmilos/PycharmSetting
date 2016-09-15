import copy
import os
import re

import sys

import signal
from bunch import Bunch
from deepsense import neptune

from ml_utils import Tee, header, start_timer, elapsed_time_mins
from neptune_utils import NeptuneTimeseriesObserver
from neptune_utils import TimeSeries
from saver import Saver, ExperimentSaver

class UrlTranslator(object):
    # (resource_name, env_var_name)
    r = [
        ('storage', 'NEPTUNE_STORAGE'),
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


def install_sigterm_handler():
        print 'installing sigterm handler!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1'
        def handleTerm(sign, no):
            print 'handleTerm()', sign
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, handleTerm)


class BasicNeptuneTrainer(object):
    @property
    def url_translator(self):
        return UrlTranslator()

    @classmethod
    def get_device(cls, device_arg):
        device_env = os.environ.get('DEVICE')
        gpu_id = os.environ.get('GPU_ID')

        device_id = None

        # Trying different options by priority, lowest first

        if device_env is not None:
            device_id = device_env
            print 'DEVICE env present, device_id={device_id}'.format(device_id=device_id)

        if (device_arg is not None) and (device_arg != 'none'):
            device_id = device_arg
            print 'device arg present, device_id={device_id}'.format(device_id=device_id)

        if gpu_id is not None:
            # This has highest priority because we use this on PSG to choose gpu
            device_id = '/gpu:{gpu_id}'.format(gpu_id=gpu_id)
            print 'GPU_ID env present, device_id={device_id}'.format(device_id=device_id)

        if device_id is None:
            from ml_utils import select_gpu

            device_id = '/gpu:{idx}'.format(idx=select_gpu())
            print 'will select device using select_gpu, device_id={device_id}'.format(device_id=device_id)

        return device_id

    @classmethod
    def translate_args(cls, url_translator, args):
        new_args = {}

        for arg_name in args:
            if len(arg_name) >= 4 and arg_name[-4:] == '_url':
                new_arg_name = arg_name[:-4] + '_path'
                new_args[new_arg_name] = url_translator.url_to_path(getattr(args, arg_name))

            new_args[arg_name] = getattr(args, arg_name)

        return Bunch(new_args)

    @classmethod
    def create_neptune_ctx(cls):
        from deepsense import neptune
        print 'create_neptune_ctx'
        return neptune.Context(sys.argv)

    def parse_args(self, neptune_ctx, do_url_translation=True):
        args_before_url_translation = neptune_ctx.params
        if do_url_translation:
            args = self.translate_args(self.url_translator, args_before_url_translation)
        else:
            args = args_before_url_translation
        return args


    def _initialize(self, neptune_ctx, args, create_neptune_channels_charts_fun):
        print '_initialize()'
        self.args = args
        self.neptune_ctx = neptune_ctx
        self.job = neptune_ctx.job

        if create_neptune_channels_charts_fun is not None:
            self.ts = create_neptune_channels_charts_fun(self.job)

        self.image_channel = self.job.create_channel(
            name='image_channel',
            channel_type=neptune.ChannelType.IMAGE)

        print 'dump_dir', self.neptune_ctx.dump_dir_url
        def make_url(z):
            if z[0] == '/' and z[1] != '/':
                return '/' + z
            else:
                return z

        self.exp_dir_path = self.url_translator.url_to_path(make_url(self.neptune_ctx.dump_dir_url))
        print 'exp_dir_path', self.exp_dir_path

        self.job.finalize_preparation()

        if self.exp_dir_path:
            print 'creating ExperimentSaver in ', self.exp_dir_path
            self.saver = ExperimentSaver(self.exp_dir_path)
            a = self.saver.open_file(None, 'mylog.txt')
            self.log_file, filepath = a.file, a.filepath
            self.tee_stdout = Tee(sys.stdout, self.log_file)
            self.tee_stderr = Tee(sys.stderr, self.log_file)
            sys.stdout = self.tee_stdout
            sys.stderr = self.tee_stderr

        if 'global_saver_path' in self.args:
            path = self.url_translator.url_to_path(self.args.global_saver_path)
            print 'will create global_saver in ', path
            self.global_saver = Saver(path)
        else:
            print 'NO global_saver'



    def get_create_netptune_channels_charts_fun(self, args):
        return None

    def _main(self):
        try:
            print '_main()'
            install_sigterm_handler()
            neptune_ctx = self.create_neptune_ctx()
            args = self.parse_args(neptune_ctx)
            create_neptune_channels_charts_fun = self.get_create_netptune_channels_charts_fun(args)
            print 'create_neptune_channels_chars_fun', create_neptune_channels_charts_fun
            self._initialize(neptune_ctx, args, create_neptune_channels_charts_fun)
            print 'calling main()'
            return self.main()
        except KeyboardInterrupt:
            print 'KURWA OKKKKKKKKKKKKKKKKKKK!'
            sys.stdout.flush()
            sys.stderr.flush()
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
        except Exception:
            print 'KURWA KUUUUUUUUUUUUUURWA'
            import traceback
            print traceback.format_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

        print 'KURWA No exception'
        sys.stdout.flush()
        sys.stderr.flush()

    def create_neptune_channels_charts(cls, job, channel_names, figures_schema):
        channels = {}

        ts = Bunch()
        for tup in channel_names:
            if len(tup) == 2:
                channel_name, mean_freq = tup
                fun_name = 'mean'
            else:
                channel_name, mean_freq, fun_name = tup
            print 'Will add channel', channel_name, mean_freq, fun_name

            channel = job.create_channel(
                name=channel_name,
                channel_type=neptune.ChannelType.NUMERIC,
                is_last_value_exposed=True,
                is_history_persisted=True)
            channels[channel_name] = channel

            ts.__setattr__(channel_name, TimeSeries())

            if fun_name == 'mean':
                fun = NeptuneTimeseriesObserver.mean
            elif fun_name == 'last':
                fun = NeptuneTimeseriesObserver.last
            else:
                raise RuntimeError()

            observer = NeptuneTimeseriesObserver(
                name=channel_name,
                channel=channels[channel_name],
                add_freq=mean_freq,
                fun=fun
            )
            getattr(ts, channel_name).add_add_observer(observer)


        for chart_name, d in figures_schema.iteritems():
            chart_series = {}
            for channel_name, name_on_plot in d:
                chart_series[name_on_plot] = channels[channel_name]
                print 'add to chart ', name_on_plot, ' channel ', channel_name

            job.create_chart(name=chart_name, series=chart_series)

        return ts


    def main(self):
        # TODO: Document what fiels are added during initialization
        raise RuntimeError()


class SimplestTrainer(BasicNeptuneTrainer):
    def main(self):
        print 'Hello World!'


class SimpleTFlowTrainer(BasicNeptuneTrainer):
    def get_create_netptune_channels_charts_fun(self, args):
        return None

    def main(self):
        import tensorflow as tf

        default_device = self.get_device(self.args.device)
        print 'DEFAULT_DEVICE', default_device
        print 'will create session!'
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
            )) as sess:

            self.neptune_ctx.job.properties['default_device'] = default_device

            with tf.device(default_device):
                with sess.as_default():
                    self.go(sess)


class StandardTFlowTrainer(BasicNeptuneTrainer):
    # NOTE(maciekk): complete refactoring
    def get_create_netptune_channels_charts_fun(self, args):
        project_manager_class = self.chose_project_manager()
        return project_manager_class.create_neptune_channels_charts

    def initialize(self):
        self.dataset_manager = self.chose_dataset_manager()
        self.project_manager = self.chose_project_manager(self.args, self.saver, self.global_saver, self.ts, self.dataset_manager)

    def chose_dataset_manager(self):
        from nielsen.ingredients.dataset_managers import get_dataset_manager_class
        dataset_manager_class = get_dataset_manager_class(self.args.dataset_manager_name)
        return dataset_manager_class(self.args, self.neptune_ctx)

    def chose_project_manager(self):
        raise NotImplementedError()

    def main(self):
        import tensorflow as tf

        default_device = self.get_device(self.args.device)
        print 'DEFAULT_DEVICE', default_device
        print 'will create session!'
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
            )) as sess:
            self.neptune_ctx.job.properties['default_device'] = default_device
            with tf.device(default_device):
                self.initialize()
                with sess.as_default():
                    self.project_manager.initialize()

                    if self.args.do_training:
                        self.do_training()
                    else:
                        print 'Will not be doing training!'

    def do_training(self):
        for epoch_idx in xrange(self.args.n_epochs):
            print header('starting epoch ' + str(epoch_idx))

            if self.args.do_train:
                train_timer = start_timer()
                train_epoch_res = self.project_manager.do_train_epoch(epoch_idx, self.args.synchronous)
                if hasattr(self.ts, 'train_epoch_time'):
                    self.ts.train_epoch_time.add(elapsed_time_mins(train_timer))

            if self.args.do_valid:
                if epoch_idx % self.args.valid_freq_epochs == 0:
                    valid_timer = start_timer()
                    valid_epoch_res = self.project_manager.do_valid_epoch(epoch_idx, self.args.synchronous)

                    if hasattr(self.ts, 'valid_epoch_time'):
                        self.ts.valid_epoch_time.add(elapsed_time_mins(valid_timer))
                else:
                    print 'Ommiting validation in epoch {epoch_idx}'.format(epoch_idx=epoch_idx)

            # epoch_data = EpochData(train_loss=train_epoch_res.get('loss'),
            #                        train_cost=train_epoch_res.get('cost'),
            #                        train_accuracy=train_epoch_res.get('accuracy'),
            #                        valid_loss=valid_epoch_res.get('loss'),
            #                        valid_cost=valid_epoch_res.get('cost'),
            #                        valid_accuracy=valid_epoch_res.get('accuracy'),
            #                        epoch_params=self.url_translator.path_to_url(
            #                            valid_epoch_res.get('epoch_checkpoint_path')
            #                        ),
            #                        model_path=self.url_translator.path_to_url(
            #                            valid_epoch_res.get('epoch_checkpoint_path')
            #                        ))

            #self.exp.add_epoch_data(epoch_data.encode())
            if epoch_idx > 0 and (epoch_idx % self.args.save_variables_epochs == 0):
                self.project_manager.save_variables(epoch_idx=epoch_idx)

            print header('ending epoch ' + str(epoch_idx))

