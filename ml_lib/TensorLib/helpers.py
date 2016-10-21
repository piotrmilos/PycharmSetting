from collections import OrderedDict

from bunch import Bunch
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


def _get_variable(name, shape, initializer, l2_reg=None):
    var = tf.get_variable(name, shape, initializer=initializer)

    if l2_reg is not None:
        tf.add_to_collection('l2_reg', (var, l2_reg))

    return var

def conv_batch_normalization(x, eps=1e-6):
    n_channels = x.get_shape()[3]
    gamma = _get_variable('gamma', shape=(n_channels,),
                    initializer=tf.constant_initializer(1.0), l2_reg=None)
    beta = _get_variable('beta', shape=(n_channels,),
                    initializer=tf.constant_initializer(1.0), l2_reg=None)
    mean = tf.reduce_mean(x, [0, 1, 2], keep_dims=True)
    x_centered = x - mean
    var = tf.reduce_mean(tf.square(x_centered), [0, 1, 2], keep_dims=True)

    x_hat = tf.div(x_centered, tf.sqrt(var + eps))

    # dimshuffle 'x', 'x', 'x', 0
    gamma_prim = tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma, 0), 0), 0)
    beta_prim = tf.expand_dims(tf.expand_dims(tf.expand_dims(beta, 0), 0), 0)
    output = gamma_prim * x_hat + beta_prim
    return output



# def batch_normalization(x, eps=1e-6):
#     n_channels = x.get_shape()[3]
#     gamma = _get_variable('gamma', shape=(n_channels,),
#                     initializer=tf.constant_initializer(1.0), l2_reg=None)
#     beta = _get_variable('beta', shape=(n_channels,),
#                     initializer=tf.constant_initializer(1.0), l2_reg=None)
#     mean = tf.reduce_mean(x, [0, 1, 2], keep_dims=True)
#     x_centered = x - mean
#     var = tf.reduce_mean(tf.square(x_centered), [0, 1, 2], keep_dims=True)
#
#     x_hat = tf.div(x_centered, tf.sqrt(var + eps))
#
#     # dimshuffle 'x', 'x', 'x', 0
#     gamma_prim = tf.expand_dims(tf.expand_dims(tf.expand_dims(gamma, 0), 0), 0)
#     beta_prim = tf.expand_dims(tf.expand_dims(tf.expand_dims(beta, 0), 0), 0)
#     output = gamma_prim * x_hat + beta_prim
#     return output


def get_loss(logits, labels, mb_size=32, n_classes=447):

    sparse_labels = tf.reshape(labels, [-1, 1])
    indices = tf.reshape(tf.range(0, mb_size), [mb_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated,
                                        [mb_size, n_classes],
                                        1.0, 0.0)

    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, dense_labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return Bunch(mean_loss=cross_entropy_mean, loss=cross_entropy)


# NOTE: copied from http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
class ConvolutionalBatchNormalizer(object):
    """Helper class that groups the normalization logic and variables.

    Use:
      ewma = tf.train.ExponentialMovingAverage(decay=0.99)
      bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)
      update_assignments = bn.get_assigner()
      x = bn.normalize(y, train=training?)
      (the output x will be batch-normalized).
    """
    def __init__(self, depth, epsilon, ewma_trainer, scale_after_norm,
                 beta_init=0.0, gamma_init=1.0):
        self.mean = tf.Variable(tf.constant(0.0, shape=[depth]),
                                trainable=False, name='bn_mean')
        self.variance = tf.Variable(tf.constant(1.0, shape=[depth]),
                                    trainable=False, name='bn_variance')

        self.beta = tf.Variable(tf.constant(beta_init, shape=[depth]), name='bn_beta', trainable=True)
        self.gamma = tf.Variable(tf.constant(gamma_init, shape=[depth]), name='bn_gamma', trainable=True)
        self.ewma_trainer = ewma_trainer
        self.epsilon = epsilon
        self.scale_after_norm = scale_after_norm

    def get_assigner(self):
        """Returns an EWMA apply op that must be invoked after optimization."""
        return self.ewma_trainer.apply([self.mean, self.variance])

    def normalize(self, x, train=True):
        """Returns a batch-normalized version of x."""
        if train:
            # dimension 3 is the filters
            mean, variance = tf.nn.moments(x, [0, 1, 2])
            assign_mean = self.mean.assign(mean)
            assign_variance = self.variance.assign(variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_norm_with_global_normalization(
                    x, mean, variance, self.beta, self.gamma,
                    self.epsilon, self.scale_after_norm)
        else:
            mean = self.ewma_trainer.average(self.mean)
            variance = self.ewma_trainer.average(self.variance)
            local_beta = tf.identity(self.beta)
            local_gamma = tf.identity(self.gamma)
            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, local_beta, local_gamma,
                self.epsilon, self.scale_after_norm)


def create_variable(name, shape, initializer=None, trainable=True):
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def my_reshape(x, shape):
    print 'my_reshape', shape
    if isinstance(shape, list) or isinstance(shape, tuple):
        shape = tf.pack(shape)

    res = tf.reshape(x, shape)
    return res


def get_rnn_cell(cell_type, hidden_size, name='rnn_cell'):
    with tf.variable_scope(name):
        if cell_type == 'gru':
            single_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
        elif cell_type == 'lstm':
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        else:
            raise RuntimeError()
        return single_cell

import numpy as np

def desc_variable(v):
    mean = tf.reduce_mean(v).eval()
    return 'Tensor({name}, shape={shape}, dtype={dtype}, mean={mean}, nof_params = {nof_params})'.format(
        name=v.name,
        shape=str(v.get_shape()),
        dtype=str(v.dtype),
        mean=str(mean),
        nof_params=np.prod(map(int, v.get_shape()))
    )


def flatten(ts):
    sh_ts = tf.shape(ts)
    sh = ts.get_shape()

    sh_rest = tf.Dimension(1)
    for a in sh[1:]:
        sh_rest *= a
    print 'kurwa'
    print sh
    print sh[0], sh_rest
    res = my_reshape(ts, [sh_ts[0], sh_rest])
    print res.get_shape()
    return res


def cross_entropy(y_res, y_corr):
    return -y_corr * tf.log(y_res)


def apply_fc(res, weigths, biases):
    return tf.matmul(res, weigths) + biases


def clip(a, min_v=None, max_v=None):
    if min_v is not None:
        a = tf.maximum(a, min_v)
    if max_v is not None:
        a = tf.minimum(a, max_v)
    return a


def general_matmul(a, a_ndim, b, b_ndim):
    # a is D1 x D2 ... x D
    # b is D x ... D_n
   # a_ndim = tf.size(tf.shape(a))

    print a_ndim
    a_shape = tf.shape(a)
    b_shape = tf.shape(b)

    #return a_shape[a_ndim - 1]
    #return a_reshaped

    # TODO: how to get last elem of tensor?
    a_reshaped = my_reshape(a, [-1, a_shape[a_ndim - 1]])
    b_reshaped = my_reshape(b, [b_shape[0], -1])
    c = tf.matmul(a_reshaped, b_reshaped)
    new_shape = tf.concat(0, [a_shape[:a_ndim - 1], b_shape[1:]])
    return my_reshape(c, new_shape)


def add_at_position(ts, pos, v):
    # TODO: what if pos is a tensor, then convert_to_tensor will fail
    # how to convert list of tensors to 1+dim tensor

    print 'add at position', ts, pos, v
    v = tf.cast(v, ts.dtype)
    v = tf.pack([v])

    print tf.convert_to_tensor([pos])
    print v
    print tf.shape(ts)

    # TODO: why the fuck I have to convert them to int64?
    delta = tf.SparseTensor(tf.convert_to_tensor([pos], dtype=tf.int64),
                            v,
                            tf.cast(tf.shape(ts), tf.int64))
    result = ts + tf.sparse_tensor_to_dense(delta)
    return result


def shift_right(a, n_dim, dim, k, cyclic=False):
    if cyclic:
        raise NotImplementedError('Only cyclic=False is implemented')

    shape = tf.shape(a)

    begin = n_dim * [0]
    size = add_at_position(tf.shape(a), (dim,), -k)
    size_resid = add_at_position(tf.shape(a), (dim,), -shape[dim] + k)

    begin = tf.Print(begin, [begin])
    size = tf.Print(size, [size])

    slice_main = tf.slice(a, begin, size)

    # ok, now we have to add v's at the beginning of the shifted dimension
    slice_resid = tf.zeros(shape=size_resid, dtype=a.dtype)
    return tf.concat(dim, [slice_resid, slice_main])




def get_verify_finite_fun(do_verify_finite):
        def verify_finite(ts):
            if do_verify_finite:
                ts = tf.verify_tensor_all_finite(ts, msg='Tensors {name} is not ok'.format(name=ts.name))
            return ts
        return verify_finite


def softmax3(x):
    print 'softmax3', x, x.get_shape()
    sh = tf.shape(x)
    x = my_reshape(x, (sh[0] * sh[1], sh[2]))
    x = tf.nn.softmax(x)
    x = tf.reshape(x, sh)
    return x


def softmax4(x):
    print 'softmax4', x, x.get_shape()
    sh = tf.shape(x)
    x = my_reshape(x, (sh[0] * sh[1] * sh[2], sh[3]))
    x = tf.nn.softmax(x)
    x = tf.reshape(x, sh)
    return x


def softmax5(x):
    print 'softmax5', x, x.get_shape()
    sh = tf.shape(x)
    x = my_reshape(x, (sh[0] * sh[1] * sh[2] * sh[3], sh[4]))
    x = tf.nn.softmax(x)
    x = tf.reshape(x, sh)
    return x


def print_shape(ts):
    print 'Tensor {name}, static_shape {shape}'.format(
        name=ts.name,
        shape=ts.get_shape()
    )


def create_apply_fc(x, n_output, n_input=None, activation=tf.nn.relu, name='fc',
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                    biases_initializer=tf.constant_initializer(0.0),
                    do_batch_norm=False):
    # x is MB x D
    D = x.get_shape()[1]

    if n_input is not None:
        D = n_input

    with tf.variable_scope(name):
        weights = create_variable('weights', [D, n_output],
                                      initializer=weights_initializer)

        biases = create_variable('biases', [n_output],
                                 initializer=biases_initializer)

        res = tf.matmul(x, weights)

        if biases_initializer is not None:
            res += biases

        if do_batch_norm:
            res = batch_norm(res)

        if activation is not None:
            res = activation(res)

        return res