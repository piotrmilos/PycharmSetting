import copy

import tensorflow as tf

from TensorLib.helpers import create_variable, ConvolutionalBatchNormalizer


def get_default_conv_creator():
    return ConvCreator(filter_size=(3, 3),
                       filter_stride=(1, 1),
                       pooling_size=(2, 2),
                       pooling_stride=(2, 2),
                       do_batch_normalization=True,
                       activation_fun=tf.nn.relu)


def map_update_dict(conv_creators, dicts):
    return map(lambda conv_creator, d: conv_creator.update_by_dict(d), zip(conv_creators, dicts))


def map_update_pair(conv_creator, l):
    res = []
    for n_filters, do_pooling in l:
        conv_creator_copy = copy.copy(conv_creator)
        conv_creator_copy.n_filters = n_filters
        conv_creator_copy.do_pooling = do_pooling
        res.append(conv_creator_copy)
    return res


def get_n_filters(res):
    n_filters = res.get_shape()[3]
    return n_filters


class ConvCreator(object):
    # TODO: Make setters/getters, but automatic not manual
    # TODO(maciek): add sth like, making sure attributes are set, fully configured
    def __init__(self, n_filters=None,
                 filter_size=None,
                 filter_stride=None,
                 pooling_size=None,
                 pooling_stride=None,
                 activation_fun=None,
                 do_conv=None,
                 do_pooling=None,
                 do_batch_normalization=None,
                 name=None,
                 filter_initializer=tf.truncated_normal_initializer(stddev=0.0001),
                 bias_initializer=tf.constant_initializer(0.0),
                 trainable=None,
                 l2_reg_mult=None,
                 name_suffix='no_name_suffix'
                 ):


        self.n_filters = n_filters
        self.filter_size = filter_size
        self.filter_stride = filter_stride
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.activation_fun = activation_fun
        self.do_pooling = do_pooling
        self.do_batch_normalization = do_batch_normalization
        self.name = name
        self.filter_initializer = filter_initializer
        self.bias_initializer = bias_initializer
        self.do_conv = do_conv
        self.trainable = trainable
        self.l2_reg_mult = l2_reg_mult
        self.name_suffix = name_suffix

    def update(self, **d):
        conv_creator_copy = copy.copy(self)
        for key, value in d.iteritems():
            setattr(conv_creator_copy, key, value)
        return conv_creator_copy

    def assert_set(self, *args):
        for arg in args:
            if getattr(self, arg) is None:
                raise RuntimeError(arg + ' should be set')

    def apply(self, res):
        # these we will return
        features = res

        prev_channels = get_n_filters(res)
        n_filters = prev_channels
        print 'prev_channels', prev_channels

        self.assert_set('do_conv')
        if self.do_conv:
            with tf.variable_scope(self.name):

                self.assert_set('filter_size', 'n_filters', 'filter_initializer', 'bias_initializer',
                                'filter_stride', 'name_suffix', 'trainable', 'l2_reg_mult')
                filters = create_variable('filters',
                                          [self.filter_size[0], self.filter_size[1], prev_channels, self.n_filters],
                                          initializer=self.filter_initializer, trainable=self.trainable)

                biases = create_variable('biases',
                                         [self.n_filters],
                                         initializer=self.bias_initializer, trainable=self.trainable)

                # NOTE(maciek): this should rather be a set, not a collection
                tf.add_to_collection('l2_reg', (filters, self.l2_reg_mult))
                tf.add_to_collection('l2_reg', (biases, self.l2_reg_mult))

            ########## convlution + biases ##########
            features = tf.nn.conv2d(res, filters,
                                    strides=[1, self.filter_stride[0], self.filter_stride[1], 1],
                                    padding='SAME',
                                    name='conv_{name_suffix}'.format(name_suffix=self.name_suffix))
            features += biases

            ########## batch normalization ##########
            self.assert_set('do_batch_normalization')
            if self.do_batch_normalization:
                self.assert_set('n_filters')
                ewma = tf.train.ExponentialMovingAverage(decay=0.99)
                bn = ConvolutionalBatchNormalizer(self.n_filters, 0.001, ewma, True)
                # update_assignments = bn.get_assigner()

                features = bn.normalize(features, train=True)

            ########## activation ##########
            self.assert_set('activation_fun')
            if self.activation_fun == 'identity':
                pass
            else:
                features = self.activation_fun(features)
        else:
            print 'NO conv'

        ########## pooling ##########

        self.assert_set('do_pooling')
        if self.do_pooling:
            self.assert_set('pooling_size', 'pooling_stride')
            features = tf.nn.max_pool(features,
                                      ksize=[1, self.pooling_size[0], self.pooling_size[1], 1],
                                      strides=[1, self.pooling_stride[0], self.pooling_stride[1], 1],
                                      padding='SAME', name='pool_{name_suffix}'.format(name_suffix=self.name_suffix))

        print 'after conf', self.name_suffix, features.get_shape()
        return {'res': features}


def create_conv_tower(input, conv_creators, automatic_naming=True):
    res = input
    outputs = []
    for conv_idx, conv_creator in enumerate(conv_creators):
        conv_creator.update(name_suffix=str(conv_idx))
        # features shape = [batch, in_height, in_width, in_channels]`
        # filters shape [filter_height, filter_width, in_channels, out_channels]`
        if automatic_naming:
            with tf.variable_scope('conv' + str(conv_idx)):
                res = conv_creator.apply(res)['res']
        else:
            res = conv_creator.apply(res)['res']

        outputs.append(res)

    return {'res': res, 'outputs': outputs}