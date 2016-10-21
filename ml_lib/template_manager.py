import tensorflow as tf
from bunch import Bunch

from TensorLib.helpers import desc_variable
from base_project_manager import BaseProjectManager
from ml_utils import unpack
from neptune_utils import TimeSeries, NeptuneTimeseriesObserver
from nielsen.ingredients.models import get_model_create


class TemplateManager(BaseProjectManager):
    def create_neptune_channels_charts(self):
        raise NotImplementedError()

    @classmethod
    def create_neptune_channels_charts_(cls, job, channels_figures):
        channel_names = channels_figures.channel_names
        figures_schema = channels_figures.figures_schema

        from deepsense import neptune
        channels = {}
        print 'channel_names', channel_names

        ts = Bunch()
        for channel_name, mean_freq in channel_names:
            channel = job.create_channel(
                name=channel_name,
                channel_type=neptune.ChannelType.NUMERIC,
                is_last_value_exposed=True,
                is_history_persisted=True)

            channels[channel_name] = channel

            ts.__setattr__(channel_name, TimeSeries())
            observer = NeptuneTimeseriesObserver(
                    name=channel_name,
                    channel=channels[channel_name], add_freq=mean_freq)
            getattr(ts, channel_name).add_add_observer(observer)


        for chart_name, d in figures_schema.iteritems():
            chart_series = {}
            for channel_name, name_on_plot in d:
                chart_series[name_on_plot] = channels[channel_name]
                print 'add to chart ', name_on_plot, ' channel ', channel_name

            job.create_chart(name=chart_name, series=chart_series)



        return ts

    def __init__(self, args, saver, global_saver, ts, dataset_manager, trainer=None):
        # NOTE: I think that the arguments that this manager uses should be declared here
        self.args = args
        self.ts = ts
        self.saver = saver
        self.global_saver = global_saver
        self.dataset_manager = dataset_manager
        self.trainer = trainer


    def initialize_model_and_functions(self):
        raise NotImplementedError

    def initialize_model_and_functions_helper(self, create_d):
        create = get_model_create(self.args.model_name)
        res, functions = unpack(create(create_d), 'res', 'functions')

        variables_desc = self.variables_desc()
        nof_params = self.get_nof_params()
        self.set_property('nof_params', nof_params)
        self.set_property('variables_desc', variables_desc)


        all_variables = tf.all_variables()
        all_trainable_variables = tf.trainable_variables()

        self.create_variable_savers()

        initialize_op = tf.initialize_all_variables()
        initialize_op.run()
        print 'Variables initialized.'

        if self.args.restore_checkpoint_path:
            print 'restore_checkpoint_path', self.args.restore_checkpoint_path, type(self.args.restore_checkpoint_path)
            self.restore_variables(self.args.restore_checkpoint_path)

        self.eval_function = functions.get('eval_function')
        self.train_function = functions.get('train_function')
        self.valid_function = functions.get('valid_function')


        print 'ALL VARIABLES:'
        print '\n'.join(map(desc_variable, all_variables))
        print ''

        print 'TRAINABLE VARIABLES:'
        print '\n'.join(map(desc_variable, all_trainable_variables))
        print ''

        print 'L2_REG_MULT:'
        print '\n'.join(map(lambda a: (desc_variable(a[0]) + ', l2_reg_mult = ' + str(a[1])), tf.get_collection('l2_reg')))
        print ''

        print 'NOF_PARAMS', nof_params

        print res


    @classmethod
    def create_neptune_channels_charts(cls, job):
        # FREQ1 = 20
        # channel_names = [
        #             'train_cost_mb',
        #             ]
        # figures_schema = OrderedDict([
        #     ('train: batches', [
        #         ('train_cost_mb', 'cost', FREQ1),
        #     ]),
        # ])
        # return cls.create_neptune_channels_charts_(job, channel_names, figures_schema)

        raise NotImplementedError()


    def do_train_epoch(self, epoch_idx):
        # train_recipes = self.dataset_manager.create_train_recipes()
        # print 'len(train_recipes)', len(train_recipes)
        #
        # train_iterator = self.dataset_manager.get_iterator(train_recipes, mb_size=5, partial_batches=True,
        #                                                    synchronous=False)
        #
        #
        # for mb_idx, b in enumerate(train_iterator):
        #     batch = b.batch
        #     print b
        raise NotImplementedError()

    def do_valid_epoch(self, epoch_idx):
        raise NotImplementedError()