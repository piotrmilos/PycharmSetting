from __future__ import absolute_import
from collections import OrderedDict

import copy

from bunch import Bunch
import cPickle
from theano.sandbox.cuda import dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from TheanoLib.init import Normal, Uniform, Constant


# I tried CURAND, but it does not speed up things, and is not portable.
#from theano.sandbox.cuda.rng_curand import CURAND_RandomStreams as RandomStreams

import numpy as np
import theano.tensor as T
import theano
from TheanoLib import utils
from TheanoLib.models import key_diff
from TheanoLib.utils import PrintValueOp
from ml_utils import floatX
import ml_utils

vcopy = copy.copy

def force_pair(a):
    if isinstance(a, int):
        return (a, a)
    else:
        return a


def identity(x):
    return x


def rectify(x):
    return T.maximum(x, 0.0)


relu = rectify


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def softmax(x):
    return T.nnet.softmax(x)


def softmax3(x):
    # TODO: is this fast or slow?
    sh = x.shape
    x = utils.PrintShapeOp(x, 'softmax3')
    x = T.reshape(x, (sh[0] * sh[1], sh[2]))
    x = softmax(x)
    x = T.reshape(x, sh)
    return x

def gram_matrix_conv(x):
    x = x.flatten(ndim=2)
    g = T.tensordot(x, x, axes=([1], [1]))
    return g

def gram_matrix_flat(x):
    g = T.tensordot(x, x, axes=([0], [0]))
    return g

def ftf_cost(gram_matrix, p=2):
    wd = gram_matrix - T.nlinalg.extract_diag(gram_matrix)
    return T.sum(wd ** p)


############################################

# TODO: this whole module badly needs unittests
class Param(object):
    def __init__(self, sh=None, lr=None, l2_reg=None):
        self.sh = sh
        self.lr = lr
        self.l2_reg = l2_reg

    def add_attr(self, attr_name, attr_value):
        setattr(self, attr_name, attr_value)


class Node(object):
    # input_nodes should be list.
    # we could also make them a dict in the future.

    def __init__(self, module):
        self.module = module
        self.input_nodes = []

    def add_input_node(self, node):
        self.input_nodes.append(node)

    def iter_input_nodes(self):
        return self.input_nodes

    def split(self, n):
        raise NotImplementedError

    def __str__(self):
        return 'Node(%s)' % (str(self.module),)

    __repr__ = __str__


def to_new_opt_param(param):
    if isinstance(param, Bunch):
        return param
    else:
        return Bunch(param=param, lr=None)


def eq(new_param1, new_param2):
        if new_param1.param == new_param2.param:
            return True
        else:
            return False


def eq2(param1, param2):
        if param1 == param2:
            return True
        else:
            return False


def unique(l, eq):
    res = []
    for a in l:
        ok = True
        for b in res:
            if eq(a, b):
                ok = False
                break
        if ok:
            res.append(a)
    return res


class Module(object):
    def __init__(self, name=''):
        self.name = name
        self.children = []

    # NOTE:
    # We should not declare any shared vars here, because when we call apply twice, we will get two
    # separate vars, which is not what we often want, shared vars should be created in __init__
    def apply(self, v, **kwargs):
        """
        This should be redefined for every module.
        :param input:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    # Currently we don't use it, and don't require redefinition.
    def get_output_shape_for(self, shape, **kwargs):
        raise NotImplementedError

    # get params methods, these should return all parameters
    def get_params(self):
        # INFO: This is tricky, I think the params returned by these methods should be unique
        # (if we share params across modules), but we have to make sure(for the saving to work) that the
        # params will be in the same order every time the program is run. We could use some stable sorting, by
        # name or some other constant thing.
        """
        :return: List of params (shared variables)
        """
        return self.params

    def get_opt_params(self):
        """
        :return: List of optimizable params (shared variables)
        """
        return self.opt_params

    def get_reg_params(self):
        """
        :return: List of regularizable params (shared variables)
        """

        return self.reg_params

    def all_params_info(self):
        params = list(set(self.get_params()))
        params = sorted(params, key=lambda param: param.name)
        weights = [param.get_value() for param in params]
        return self.weights_info(params, weights)

    def weights_info(self, params, weights):
        res = []
        nof_params = 0
        #hash = 0
        for param, weight in zip(params, weights):
            res.append('{} sum = {}, shape = {}, nof_params = {}'.format(
                param.name, np.sum(weight),  weight.shape, np.prod(weight.shape)
            ))

            nof_params += np.prod(weight.shape)

            #hash += utils.nphash(weight)
        print type(nof_params)
        res.append('Nof params in the model = {}'.format(str(nof_params)))
        return Bunch(desc='\n'.join(res), nof_params=int(nof_params))

    def save_state_new(self, filepath):
        print('Saving model to %s' % (filepath,))
        params = self.get_params()
        params = unique(params, eq2)
        names = [param.name for param in params]

        if len(names) != len(set(names)):
            raise RuntimeError('All param names should be unique.')

        print names
        state_to_save = [Bunch(value=param.get_value(), name=param.name) for param in params]
        cPickle.dump(state_to_save, file(filepath, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        print('done.')

    def load_state_new(self, filepath):
        print('Loading params from %s' % (filepath,))
        saved_state = cPickle.load(file(filepath, 'rb'))
        params = self.get_params()
        print 'len(params)', len(params), 'len(saved_state)', len(saved_state)
        saved_state_dict = {param.name: param for param in saved_state}

        params_dict = {param.name: param for param in params}

        print 'Params in current model: ', str(params_dict.keys())
        print 'Params in saved model: ', str(saved_state_dict.keys())

        print 'Params not present in saved model:', str(key_diff(params_dict, saved_state_dict))
        print 'Params present in saved model but not in current:', str(key_diff(saved_state_dict, params_dict))

        nof_params = 0
        for name, param in params_dict.iteritems():
            saved_value = None
            if name in saved_state_dict:
                saved_value = saved_state_dict[name].value

            sum = None if saved_value is None else np.sum(saved_value)
            print name, 'sum =', sum, 'shape=', param.get_value().shape
            nof_params += np.prod(param.get_value().shape)

            param_shape = param.get_value().shape
            save_shape = saved_value.shape
            print 'shapes', 'param_shape', param_shape, 'save_shape', save_shape
            if param_shape != save_shape:
                raise RuntimeError('Saved shape different that param shape ' + name)

            if saved_value is not None:
                param.set_value(saved_value)

        print 'Nof params in the mode =', nof_params

    # We will be using similar syntax a torch's nngraph
    def __call__(self, input_nodes=[]):
        # AAA
        if not isinstance(input_nodes, list):
            input_nodes = [input_nodes]

        node = Node(module=self)
        for input_node in input_nodes:
            node.add_input_node(input_node)

        return node

    def ftf_cost(self):
        return 0.0

    # Convenience functions
    def create_shared(self, shared_name, initializer, shape):
        #print 'create shared', shape
        return theano.shared(value=initializer(shape), name=self.name + '.' + shared_name)

    def show(self, filepath):
        raise NotImplementedError()

    def post_apply(self, v, **kwargs):
        if isinstance(v, Bunch) and 'post_apply_fun' in kwargs:
            return kwargs['post_apply_fun'](self, v)
        else:
            return v

    def allocate_params(self):
        for child in self.children:
            child.allocate_params()

        return self._allocate_params()

    def initialize_params(self):
        for child in self.children:
            child.initialize_params()

        return self._initialize_params()

    def _allocate_params(self):
        pass

    def _initialize_params(self):
        pass



class Identity(Module):
    def __init__(self):
        super(Identity, self).__init__()
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        return self.post_apply(vcopy(v), **kwargs)


class Graph(Module):
    # CHECK: doeas post_apply works with Graph module?
    def dfs(self, act_node, done, topo):
        done[act_node] = True

        for input_node in act_node.iter_input_nodes():
            if input_node not in done:
                self.dfs(input_node, done, topo)

        topo.append(act_node)

    def __init__(self, input_nodes, output_nodes, name=''):
        super(Graph, self).__init__(name)
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        topo = []
        done = {}
        for node in output_nodes:
            self.dfs(node, done, topo)

        self.forward_topo = topo
        self.params, self.opt_params, self.reg_params = [], [], []

        for node in topo:
            self.params += node.module.get_params()
            self.opt_params += node.module.get_opt_params()
            self.reg_params += node.module.get_reg_params()

    def apply(self, v, **kwargs):
        output_bunches = {}

        if len(self.input_nodes) == 1:
            output_bunches[self.input_nodes[0]] = self.input_nodes[0].module.apply(v, **kwargs)
        else:
            print 'tak'
            for idx, input_node in enumerate(self.input_nodes):
                output_bunches[input_node] = input_node.module.apply(v[idx], **kwargs)
        print self.forward_topo

        for node in self.forward_topo:
            if node not in output_bunches:
                input_bunches = []
                for input_node in node.iter_input_nodes():
                    input_bunches.append(output_bunches[input_node])

                if len(input_bunches) == 1:
                    # This semantic is a little strange
                    # AAA
                    input_bunches = input_bunches[0]

                output_bunches[node] = node.module.apply(input_bunches, **kwargs)


        output = []
        for output_node in self.output_nodes:
            output.append(output_bunches[output_node])

        if len(output) == 1:
            output_v = output[0]
        else:
            output_v = output

        return self.post_apply(output_v, **kwargs)

    def __str__(self):
        res = 'Topological order:\n'
        for node in self.forward_topo:
            res += str(node.module) + '\n'
        return res


class Reshape(Module):
    def __init__(self, shape, name=''):
        super(Reshape, self).__init__(name)
        self.shape = shape
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        input = v.output
        shape = list(copy.copy(self.shape))

        input_size = T.prod(input.shape)
        output_size_not_none = T.prod(filter(lambda a: a is not None, shape))

        for i in xrange(len(shape)):
            if shape[i] is None:
                shape[i] = input_size / output_size_not_none
                print 'Inferred size for {} axis'.format(i)
        output = input.reshape(shape)

        output_v = vcopy(v)
        output_v.update(output=output)
        return self.post_apply(output_v, **kwargs)

    def __str__(self):
        d = [('name', self.name),
             ('shape', self.shape)]
        return 'Reshape ' + utils.list_desc(d)


class FanOut(Module):
    def __init__(self, n, name=''):
        super(FanOut, self).__init__(name)
        self.n = n
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        res_v_list = []
        for a in xrange(self.n):
            res_v_list.append(copy.copy(v))

        return self.post_apply(res_v_list, **kwargs)

    def __str__(self):
        d = [('name', self.name),
             ('n', self.n)]
        return 'FanOut ' + utils.list_desc(d)


class Parallel(Module):
    def __init__(self, modules, name=''):
        super( Parallel, self).__init__(name)
        self.modules = modules
        self.children = modules

    def apply(self, v_list, **kwargs):
        assert(len(v_list) == len(self.modules))
        res_v_list = []
        print len(self.modules)
        for module, v in zip(self.modules, v_list):
            print module.name
            print type(module), module
            res_v_list.append(module.apply(v, **kwargs))

        return self.post_apply(res_v_list, **kwargs)

    def get_params(self):
        s = []
        for module in self.modules:
            s += module.get_params()
        return s

    def get_opt_params(self):
        s = []
        for module in self.modules:
            s += module.get_opt_params()
        return s

    def get_reg_params(self):
        s = []
        for module in self.modules:
            s += module.get_reg_params()
        return s

    def __str__(self):
        res = 'Parallel:\n'
        for module in self.modules:
            res += str(module) + '\n\n'
        return res


class Flatten(Module):
    def __init__(self, name=''):
        super(Flatten, self).__init__(name)
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        input = v.output
        n_inputs = T.prod(input.shape[1:])
        output = input.reshape((input.shape[0], n_inputs))

        output_v = vcopy(v)
        output_v.update(output=output)
        return self.post_apply(output_v, **kwargs)

    def __str__(self):
        d = [('name', self.name)]
        return 'Flatten ' + utils.list_desc(d)


class Sequential(Module):
    def __init__(self, name='', modules=[]):
        super(Sequential, self).__init__(name)
        self.modules = []
        for module in modules:
            self.add(module)

    def add(self, module):
        self.modules.append(module)
        self.children.append(module)
        return module

    def __getitem__(self, item):
        return self.modules[item]

    def apply(self, v, **kwargs):

        for module in self.modules:
            v = module.apply(v, **kwargs)
        output_v = vcopy(v)
        return self.post_apply(output_v, **kwargs)

    def get_params(self):
        s = []
        for module in self.modules:
            s += module.get_params()
        return s

    def get_opt_params(self):
        s = []
        for module in self.modules:
            s += module.get_opt_params()
        return s

    def get_reg_params(self):
        s = []
        for module in self.modules:
            s += module.get_reg_params()
        return s

    def __str__(self):
        res = 'Sequential:\n'
        for module in self.modules:
            res += str(module) + '\n\n'
        return res

    def ftf_cost(self):
        return sum(m.ftf_cost() for m in self.modules)


class DenseMul(Module):
    def __init__(self, n_input, n_output, W_init=None, W_lr=None, b_lr=None, b_init=Constant(0.0), name='',
                 numpy_rng=None):
        if W_init is None:
            W_bound = np.sqrt(1. / n_input)
            W_init = Uniform(range=W_bound)

        super(DenseMul, self).__init__(name)
        self.n_input = n_input
        self.n_output = n_output
        self.numpy_rng = numpy_rng
        self.W_init = W_init
        self.b_init = b_init
        self.W_lr = W_lr
        self.b_lr = b_lr

    def _allocate_params(self):
        self.W = self.create_shared('W', self.W_init, (self.n_input, self.n_output))
        self.b = self.create_shared('b', self.b_init, (self.n_output,))

        self.params = [self.W, self.b]

        self.opt_params = [Bunch(param=self.W, lr=self.W_lr),
                           Bunch(param=self.b, lr=self.b_lr)]

        self.reg_params = [self.W]


    def apply(self, v, **kwargs):
        input = v.output
        output = T.dot(input, self.W) + self.b
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('name', self.name),
             ('n_input', self.n_input),
             ('n_output', self.n_output),
             ('W_init', str(self.W_init)),
             ('b_init', str(self.b_init))
        ]

        return 'DenseMul ' + utils.list_desc(d)

    def ftf_cost(self, p=2):
        g = gram_matrix_flat(self.W)
        return ftf_cost(g, p=p)


class TorchThreshold(Module):
    def __init__(self, a, b):
        super(TorchThreshold, self).__init__(name)
        self.a = a
        self.b = b
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        input = v.output
        output = T.switch(input <= self.a, self.b, input)
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('a', self.a),
             ('b', self.b)
             ]
        return 'TorchThreshold ' + utils.list_desc(d)

def gaussian_noise(shape, avg, std, seed):
    return RandomStreams(seed).normal(shape, avg=avg, std=std, dtype=ml_utils.floatX)

class Dropout(Module):
    def __init__(self, p_of_zero=0.5, name=''):
        super(Dropout, self).__init__(name)
        self.p_of_zero = p_of_zero
        self.name = name
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        """
        During training you should pass a parameter mode='train', during validation and testing
        it should be set to 'test'.
        """
        input = v.output
        #print 'mode', kwargs['mode']

        if kwargs['mode'] == 'train':
            SEED = 5

            input /= 1 - self.p_of_zero

            # NOTICE: This is probably a small BUG. We crate a RandomStream object for each dropout layer, and
            # seed it with the same SEED.
            mask_input = RandomStreams(SEED).uniform(input.shape, low=0, high=1, dtype=ml_utils.floatX)
            mask = T.switch(mask_input <= self.p_of_zero, 0.0, 1.0)
            output = input * mask

            nv = vcopy(v)
            nv.update(output=output)
            # Generating from binomial distribution is VERY slow, we do it manually above.
            #return input * RandomStreams(1, use_cuda=True).binomial(input.shape, p=(1 - self.p_of_zero), dtype='int32')
        elif kwargs['mode'] == 'test':
            nv = vcopy(v)
        else:
            raise RuntimeError()

        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('name', self.name),
            ('p_of_zero', self.p_of_zero)]
        return 'Dropout ' + utils.list_desc(d)


class GaussianNoise(Module):
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        """
        During training you should pass a parameter mode='train', during validation and testing
        it should be set to 'test'.
        """
        input = v.output
        print 'mode', kwargs['mode']

        if kwargs['mode'] == 'train':
            SEED = 5

            noise = RandomStreams(SEED).normal(input.shape, avg=0, std=self.sigma, dtype=ml_utils.floatX)
            output = input + noise
            nv = vcopy(v)
            nv.update(output=output)
        elif kwargs['mode'] == 'test':
            nv = vcopy(v)
        else:
            raise RuntimeError()

        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('sigma', self.sigma)]
        return 'GaussianNoise ' + utils.list_desc(d)


class Dense(Module):
    def __init__(self, n_input, n_output, nonlinearity, W_init=None,
                 b_init=Constant(0.0), W_lr=None, b_lr=None, name='', batch_norm=False, numpy_rng=None):
        # TODO: make this work for 2, 3 dimensional inputs
        # TODO: better initialization
        # TODO: add no_bias handling
        """
        input should be in (mb, data) order
        """
        super(Dense, self).__init__(name)
        self.seq = Sequential()
        self.seq.add(DenseMul(n_input, n_output, W_init=W_init, b_init=b_init, W_lr=W_lr, b_lr=b_lr, numpy_rng=numpy_rng, name=name + '.DenseMul'))

        if batch_norm:
            self.seq.add(BatchNormalization(shape=(n_output,), name=name + '.BN'))

        self.seq.add(ApplyActivation(nonlinearity))

        self.children = copy.copy(self.seq.modules)

    def get_params(self):
        return self.seq.get_params()

    def get_opt_params(self):
        return self.seq.get_opt_params()

    def get_reg_params(self):
        return self.seq.get_reg_params()

    def apply(self, v, **kwargs):
        nv = vcopy(self.seq.apply(v, **kwargs))
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        res = 'Dense:\n'
        for module in self.seq.modules:
            res += str(module) + '\n'
        return res

    def ftf_cost(self):
        return self.seq.ftf_cost()


class Softmax(Module):
    def __init__(self, name=''):
        super(Softmax, self).__init__(name)
        self.name = name
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        input = v.output
        if input.ndim <= 3:
            output = softmax(input)
        else:
            input_reshaped = input.reshape([T.prod(input.shape[:-1]), input.shape[-1]])
            output_reshaped = softmax(input_reshaped)
            output = output_reshaped.reshape(input.shape)

        nv = vcopy(v)
        nv.update(output=output)

        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('name', self.name)]
        return 'Softmax' + utils.list_desc(d)


class SimpleApply(Module):
    def __init__(self, fun, name=''):
        super(SimpleApply, self).__init__(name)
        self.fun = fun
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        input = v.output

        output = self.fun(input)

        nv = vcopy(v)
        nv.update(output=output)

        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('name', self.name)]
        return 'SimpleApply' + utils.list_desc(d)


class SimpleListApply(Module):
    def __init__(self, fun, name=''):
        super(SimpleListApply, self).__init__(name)
        self.fun = fun
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v_list, **kwargs):
        # WARNING: we loose info other than output, THIS IS BAD, but there is not best semantics for this
        inputs = map(lambda v: v.output, v_list)

        output = self.fun(inputs)
        nv = Bunch(output=output)

        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('name', self.name)]
        return 'SimpleListApply' + utils.list_desc(d)


class ListApplyFun(Module):
    def __init__(self, fun, name=''):
        super(ListApplyFun, self).__init__(name)
        self.fun = fun
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v_list, **kwargs):
        return self.fun(v_list)

    def __str__(self):
        d = [('name', self.name)]
        return 'ListApplyFun' + utils.list_desc(d)


class ConvBatchNormalization(Module):
    # TODO: we don't do inference/testing/validation phase properly

    def __init__(self, shape, name='conv_batch_norm', eps=1e-6):
        super(ConvBatchNormalization, self).__init__(name)
        self.eps = eps
        self.shape = shape

    def _allocate_params(self):
        self.gamma = self.create_shared(shape=self.shape, initializer=Constant(1.0), shared_name='gamma')
        self.beta = self.create_shared(shape=self.shape, initializer=Constant(0.0), shared_name='beta')

        self.opt_params = [Bunch(param=self.gamma), Bunch(param=self.beta)]
        self.params = [self.gamma, self.beta]
        # WARN: do we want to regularize it or not?
        self.reg_params = []

    def apply(self, v, **kwargs):
        input = v.output

        mean = T.mean(input, axis=(0, 2, 3), keepdims=True)
        #mean = PrintValueOp(mean, self.name + '_mean')
        x_centred = input - mean
        var = T.mean(x_centred ** 2, axis=(0, 2, 3), keepdims=True)
        #var = PrintValueOp(var, self.name + '_var')
        x_hat = x_centred / T.sqrt(var + self.eps)

        output = self.gamma.dimshuffle('x', 0, 'x', 'x') * x_hat + self.beta.dimshuffle('x', 0, 'x', 'x')
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('name', self.name)]
        return 'ConvBatchNormalization' + utils.list_desc(d)


class BatchNormalization(Module):
    # TODO: we don't do inference/testing/validation phase properly
    # we should computer means, variances on training data, save them and use in
    # inference/testing/validation.

    # TODO: regularization?
    def __init__(self, shape, alpha=0.9, name='batch_norm'):
        super(BatchNormalization, self).__init__(name)
        # TODO: initizaliation how to?
        self.shape = shape
        self.alpha = alpha


        # self.mean = self.create_shared(shape=shape, initializer=Constant(0.0),
        # shared_name='mean_const')
        #
        # self.variance = self.create_shared(shape=shape, initializer=Constant(0.0),
        #                                     shared_name='variance_const')

        # See https://gist.github.com/f0k/f1a6bd3c8585c400c190.
        # basically we want running_{mean,variance} to update self.mean, self.variance
        # self.running_mean = theano.clone(self.mean, share_inputs=False)
        # self.running_variance = theano.clone(self.variance, share_inputs=False)


        # # update the running average of means
        # self.mean_running_avg.default_update = alpha * self.mean_running_avg + (1 - alpha) * mean
        #
        # # update the running average of variances
        # self.variance_running_abg.default_update = alpha * self.variance_running_abg + (1 - alpha) * variance
        #
        # # NOTE: this is taken from Lasagne, default_update is not well documented, not sure if it is guaranteed
        # # to work.
        #
        # # and include them in the graph so their default updates will be
        # # applied (although the expressions will be optimized away later)
        # mean += 0 * self.mean_running_avg
        # variance += 0 * self.variance_running_abg


    def _allocate_params(self):
        self.gamma = self.create_shared(shape=self.shape, initializer=Constant(1.0), shared_name='normalization_gamma')
        self.beta = self.create_shared(shape=self.shape, initializer=Constant(0.0), shared_name='normalization_beta')
        self.opt_params = [self.gamma, self.beta]
        self.params = self.opt_params
        self.reg_params = [self.gamma]

    def apply(self, v, **kwargs):
        input = v.output
        mean = T.mean(input, axis=0)
        input_zero_mean = input - mean
        variance = T.mean(input_zero_mean ** 2, axis=0)

        # TODO: eps what value?
        eps = 0.00000001
        input_zero_mean_stdev_one = input_zero_mean / T.sqrt(variance + eps)

        output = self.gamma * input_zero_mean_stdev_one + self.beta
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('name', self.name)]
        return 'BatchNormalization' + utils.list_desc(d)


class ApplyActivation(Module):
    def __init__(self, activation_function, name=''):
        super(ApplyActivation, self).__init__(name)
        self.params, self.opt_params, self.reg_params = [], [], []
        self.activation_function = activation_function

    def apply(self, v, **kwargs):
        input = v.output
        input = utils.PrintShapeOp(input, 'apply')
        output = self.activation_function(input)
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [
            ('name', self.name),
            ('activation_function', self.activation_function.__name__),
        ]
        return 'ApplyActivation ' + utils.list_desc(d)


class MaxPooling(Module):
    def __init__(self, input_height, input_width, pooling_size, pooling_stride, name=''):
        super(MaxPooling, self).__init__(name)
        self.params, self.opt_params, self.reg_params = [], [], []

        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.input_height = input_height
        self.input_width = input_width

        if pooling_stride > pooling_size:
            raise NotImplementedError("MaxPooling only supports "
                                      "stride <= pool_size.")

    def apply(self, v, **kwargs):
        input = v.output
        #input = utils.PrintShapeOp(input, 'pool')

        input_shuffled = input.dimshuffle(1, 2, 3, 0)
        from pylearn2.sandbox.cuda_convnet.pool import MaxPool
        pool_op = MaxPool(ds=self.pooling_size, stride=self.pooling_stride)
        contiguous_input = gpu_contiguous(input_shuffled)
        output = pool_op(contiguous_input).dimshuffle(3, 0, 1, 2)
        print self.input_height, self.pooling_size, self.pooling_stride
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def get_output_height(self):
        return int(np.ceil(float(self.input_height - self.pooling_size +
                                 self.pooling_stride) / self.pooling_stride))

    def get_output_width(self):
        return int(np.ceil(float(self.input_width - self.pooling_size +
                             self.pooling_stride) / self.pooling_stride))

    def __str__(self):
        d = [
            ('name', self.name),
            ('pooling_size', self.pooling_size),
            ('pooling_stride', self.pooling_stride),
            ('act_h', self.get_output_height()),
            ('act_w', self.get_output_width())]

        return 'MaxPooling ' + utils.list_desc(d)


class CudaConv2d(Module):
    # TODO: add regularization?

    def __str__(self):
        d = [
            ('name', self.name),
            ('n_input_channels', self.n_input_channels),
            ('n_filters', self.n_filters),
            ('kernel_size', self.kernel_size),
            ('kernel_stride', self.kernel_stride),
            ('padding', self.padding),
            ('act_h', self.get_output_height()),
            ('act_w', self.get_output_width()),
            ('filter_init', self.filter_init),
            ('filter_bias_init', self.filter_init_bias),
            ('memory_est', self.get_output_height() * self.get_output_width() * self.n_filters),
            ]

        return 'CudaConv2d ' + utils.list_desc(d)

    def __init__(self, input_height, input_width, n_input_channels, n_filters, kernel_size,
                 kernel_stride, padding, filter_l2_coeff=None, filter_bias_l2_coeff=None,
                 filter_init=Uniform(range=0.1), filter_bias_init=Constant(0.0),
                 partial_sum=1, name='',
                 numpy_rng=None, untie_biases=False):
        """
        input tensor should be in bc01 order (batch, channel, dim0, dim1)
        filters shared variable should be (filters, channels in previous module, height, width)
        """
        super(CudaConv2d, self).__init__(name)

        assert (untie_biases is False)

        self.numpy_rng = numpy_rng
        self.partial_sum = partial_sum
        self.n_filters = n_filters
        self.n_input_channels = n_input_channels
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.filter_init = filter_init
        self.filter_init_bias = filter_bias_init
        self.filter_l2_coeff = filter_l2_coeff
        self.filter_bias_l2_coeff = filter_bias_l2_coeff
        self.input_height = input_height
        self.input_width = input_width
        self.padding = padding

        # TODO: how to properly initialize filters
        # INFO: importance of initialization, when trying range = 0.01 then net does not learn at all!!!

        self.filters = self.create_shared('conv_filters', filter_init,
                                          (n_filters, n_input_channels, kernel_size, kernel_size))

        # TODO: untie biases
        self.filters_bias = self.create_shared('conv_filter_bias', filter_bias_init, (n_filters,))

        self.params = [self.filters, self.filters_bias]
        self.opt_params = self.params
        self.reg_params = [self.filters]


    def get_output_height(self):
        return int(np.ceil(float(2 * self.padding + self.input_height - self.kernel_size) / self.kernel_stride + 1))

    def get_output_width(self):
        return int(np.ceil(float(2 * self.padding + self.input_width - self.kernel_size) / self.kernel_stride + 1))

    # NOTICE:
    # There seems to be some problem in lasagne handling cuda-convnet
    #
    #
    # def get_output_height(self):
    #     return conv_output_length(self.input_height, self.kernel_size, self.kernel_stride, 'pad', self.padding)
    #
    # def get_output_width(self):
    #     return conv_output_length(self.input_width, self.kernel_size, self.kernel_stride, 'pad', self.padding)

    def get_output_filters(self):
        return self.n_filters

    def apply(self, v, **kwargs):
        input = v.output

        #input = utils.PrintShapeOp(input, 'conv')
         # See http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
        # for further info about what follows.
        # See cuda-convnet for info about partial_sum
        from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
        conv_op = FilterActs(stride=self.kernel_stride, pad=self.padding, partial_sum=self.partial_sum)

        input_shuffled = input.dimshuffle(1, 2, 3, 0)
        filters_shuffled = self.filters.dimshuffle(1, 2, 3, 0)
        contiguous_input = gpu_contiguous(input_shuffled)
        contiguous_filters = gpu_contiguous(filters_shuffled)

        # out_shuffled is in channels, height, width, mb order
        out_shuffled = conv_op(contiguous_input, contiguous_filters)
        out_shuffled += self.filters_bias.dimshuffle(0, 'x', 'x', 'x')

        # unshuffling
        output = out_shuffled.dimshuffle(3, 0, 1, 2)

        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    # def show(self, filepath=None):
    #     filters = np.copy(self.filters.get_value())
    #     fshape = filters.shape
    #     print fshape
    #     res = np.zeros(shape=(fshape[0] * fshape[1], fshape[2] * fshape[3]))
    #     for a in xrange(fshape[0]):
    #         for b in xrange(fshape[1]):
    #             res[a * fshape[1] + b, :] = filters[a, b, ...].flatten('C')
    #
    #     if fshape[1] == 1:
    #         assert (fshape[0] % 4 == 0)
    #         img = deeplearning_utils.tile_raster_images(res,
    #                                                     img_shape=(self.kernel_size, self.kernel_size),
    #                                                     tile_shape=(4, fshape[0] / 4),
    #                                                     tile_spacing=(1, 1))
    #
    #     else:
    #         img = deeplearning_utils.tile_raster_images(res,
    #                                                     img_shape=(self.kernel_size, self.kernel_size),
    #                                                     tile_shape=(fshape[0], fshape[1]),
    #                                                     tile_spacing=(1, 1))
    #
    #     if filepath is None:
    #         utils.plot_image(img)
    #     else:
    #         utils.plot_image_to_file(img, filepath)
    #
    # def show(self, filepath=None):
    #     filters = np.copy(self.filters.get_value())
    #
    #     fshape = filters.shape
    #     print fshape
    #     res = np.zeros(shape=(fshape[0], 3, fshape[2] * fshape[3]))
    #
    #     for a in xrange(fshape[0]):
    #         for b in xrange(0, 3):
    #             res[a, b, :] = filters[a, b, ...].flatten('C')
    #
    #     img = deeplearning_utils.tile_raster_images((res[:, 0, ...], res[:, 1, ...], res[:, 2, ...], None),
    #                                                 img_shape=(self.kernel_size, self.kernel_size),
    #                                                 tile_shape=(4, fshape[0] / 4),
    #                                                 tile_spacing=(1, 1))
    #     print img
    #
    #     if filepath is None:
    #         utils.plot_image(img)
    #     else:
    #         utils.plot_image_to_file(img, filepath)


class ActivationMonitoring(Module):
    @classmethod
    def get_buckets(cls):
        eps = 0.00001
        # NOTICE: change for tanh to include (-inf, 0)
        return [
            # The intervals are left-closed, right open i.e. [a, b)
            (0, eps),
            (eps, 0.2),
            (0.2, 0.4),
            (0.4, 0.6),
            (0.6, 0.8),
            (0.8, 1 - eps),
            (1 - eps, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 999)
        ]
        return buckets

    def __init__(self, name='monitor'):
        super(ActivationMonitoring, self).__init__(name)
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        input = v.output

        z = T.mean(input)
        stdev = T.std(input)

        nv = vcopy(v)
        histogram = []
        buckets = self.get_buckets()
        for beg, end in buckets:
            a = T.ge(input, beg)
            b = T.lt(input, end)
            percent = T.sum(a * b) / T.prod(input.shape).astype(floatX)
            histogram.append(percent)

        r = {'name': self.name, 'mean': z, 'stdev': stdev, 'histogram': histogram}
        if 'activation_monitoring' in nv:
            nv.activation_monitoring.append(r)
        else:
            nv.activation_monitoring = [r]
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [
            ('name', self.name),
        ]
        return 'Monitoring ' + utils.list_desc(d)

def doit(image_size, padding, filter_size, stride):
    return (image_size + 2 * padding - filter_size) / stride + 1


class CudaConv2dCudnn(Module):
    # !!!!HANDLE padding = valid, cudnn allows some shapes to be 1 tuple

    def __str__(self):
        d = [
            ('name', self.name),
            ('n_input_channels', self.n_input_channels),
            ('n_filters', self.n_filters),
            ('kernel_size', self.kernel_size),
            ('kernel_stride', self.kernel_stride),
            ('padding', self.padding),
            ('act_h', self.get_output_height()),
            ('act_w', self.get_output_width()),
            ('filter_init', self.filter_init),
            ('filter_bias_init', self.filter_bias_init),
            ('workmem', self.workmem)]

        return 'CudaConv2dCudnn ' + utils.list_desc(d)

    def __init__(self, input_height, input_width, n_input_channels, n_filters, kernel_size,
                 kernel_stride, padding, workmem='none',
                 filter_init=Uniform(range=0.1), filter_bias_init=Constant(0.0),
                 partial_sum=1, name='',
                 numpy_rng=None, untie_biases=False):
        """
        input tensor should be in bc01 order (batch, channel, dim0, dim1)
        filters shared variable should be (filters, channels in previous module, height, width)
        """
        super(CudaConv2dCudnn, self).__init__(name)

        self.numpy_rng = numpy_rng
        self.partial_sum = partial_sum
        self.n_filters = n_filters
        self.n_input_channels = n_input_channels
        self.kernel_size = force_pair(kernel_size)
        self.kernel_stride = force_pair(kernel_stride)
        self.filter_init = filter_init
        self.filter_bias_init = filter_bias_init
        self.input_height = input_height
        self.input_width = input_width
        self.padding = force_pair(padding)
        self.workmem = workmem

        self.untie_biases = untie_biases
        if self.untie_biases is True:
            self.untie_biases = 1


    def _allocate_params(self):
        print 'allocate', self.name
        # TODO: how to properly initialize filters
        # INFO: importance of initialization, when trying range = 0.01 then net does not learn at all!!!
        print 'FUCK', self.get_output_height(), self.get_output_width(), self.get_output_filters()
        print 'n_filters' , self.n_filters
        print 'n_input_channels', self.n_input_channels
        print 'kernel_size', self.kernel_size
        print 'JEBBBBBBBBANY', (self.n_filters, self.n_input_channels, self.kernel_size[0], self.kernel_size[1])
        self.filters = self.create_shared('conv_filters', self.filter_init,
                                          (self.n_filters, self.n_input_channels, self.kernel_size[0], self.kernel_size[1]))

        # TODO: untie biases
        if self.untie_biases is False:
            self.filters_bias = self.create_shared('conv_filter_bias', self.filter_bias_init, (self.n_filters,))
        elif isinstance(self.untie_biases, int):
            output_height = self.get_output_height()
            output_width = self.get_output_width()
            if output_height % self.untie_biases != 0 or output_width % self.untie_biases:
                raise RuntimeError()

            shape = (self.n_filters, output_height / self.untie_biases, output_width / self.untie_biases)
            self.filters_bias = self.create_shared('conv_filter_bias_untied',
                                                   self.filter_bias_init,
                                                   shape
                                                   )
        else:
            raise RuntimeError()

        self.params = [self.filters, self.filters_bias]
        self.opt_params = [Bunch(param=self.filters), Bunch(param=self.filters_bias)]
        self.reg_params = [self.filters]

    def get_output_height(self):
        return doit(self.input_height, self.padding[0], self.kernel_size[0], self.kernel_stride[0])

    def get_output_width(self):
        return doit(self.input_width, self.padding[1], self.kernel_size[1], self.kernel_stride[1])

    def get_output_filters(self):
        return self.n_filters

    def apply(self, v, **kwargs):
        input = v.output

        #input = utils.PrintShapeOp(input, 'conv')
        #print 'DDDDDDDDDDDD', self.padding, self.kernel_stride
        if kwargs.get('use_cpu', False):
            assert(False)
            h_margin = self.kernel_size[0] - 1
            w_margin = self.kernel_size[1] - 1
            full_output = conv2d(input=input, filters=self.filters.dimshuffle(0, 1, 3, 2), border_mode='full')
            output = full_output[:, :, h_margin:h_margin + self.get_output_height(),
                     w_margin:w_margin + self.get_output_width()]
            output += self.filters_bias.dimshuffle('x', 0, 'x', 'x')
            output = utils.PrintShapeOp(output, self.name)
        else:
            output = dnn.dnn_conv(input, self.filters, border_mode=self.padding, subsample=self.kernel_stride)
            if self.untie_biases is False:
                output += self.filters_bias.dimshuffle('x', 0, 'x', 'x')
            else:
                repeated_bias = self.filters_bias
                repeated_bias = T.repeat(repeated_bias, self.untie_biases, axis=1)
                repeated_bias = T.repeat(repeated_bias, self.untie_biases, axis=2)
                output += repeated_bias.dimshuffle('x', 0, 1, 2)

        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def ftf_cost(self, p=2):
        g = gram_matrix_conv(self.filters)
        return ftf_cost(g, p=p)


class PoolingCudnn(Module):
    def __init__(self, input_height, input_width, pooling_size, pooling_stride, padding, mode='max', name=''):
        super(PoolingCudnn, self).__init__(name)
        self.params, self.opt_params, self.reg_params = [], [], []
        self.mode = mode

        self.pooling_size = force_pair(pooling_size)
        self.pooling_stride = force_pair(pooling_stride)
        self.padding = force_pair(padding)
        self.input_height = input_height
        self.input_width = input_width

        if pooling_stride > pooling_size:
            raise NotImplementedError("Pooling only supports "
                                      "stride <= pool_size.")

    def apply(self, v, **kwargs):
        input = v.output
        #input = utils.PrintShapeOp(input, 'pool')
        if kwargs.get('use_cpu', False):
            output = downsample.max_pool_2d(input=input, ds=self.pooling_size, st=self.pooling_stride)
            output = utils.PrintShapeOp(output, self.name)
        else:
            output = dnn.dnn_pool(img=input, mode=self.mode, ws=self.pooling_size, stride=self.pooling_stride,
                                  pad=self.padding)

        print self.input_height, self.pooling_size, self.pooling_stride
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def get_output_height(self):
        return doit(self.input_height, self.padding[0], self.pooling_size[0], self.pooling_stride[0])

    def get_output_width(self):
        return doit(self.input_width, self.padding[1], self.pooling_size[1], self.pooling_stride[1])

    def __str__(self):
        d = [
            ('name', self.name),
            ('pooling_size', self.pooling_size),
            ('pooling_stride', self.pooling_stride),
            ('act_h', self.get_output_height()),
            ('act_w', self.get_output_width()),
            ('mode', self.mode),
            ('padding', self.padding)]

        return 'PoolingCudnn ' + utils.list_desc(d)

class UnPooling(Module):
    def __init__(self, input_height, input_width, pooling_size, name=''):
        super(UnPooling, self).__init__(name)
        self.params, self.opt_params, self.reg_params = [], [], []

        self.pooling_size = force_pair(pooling_size)
        self.input_height = input_height
        self.input_width = input_width

    def apply(self, v, **kwargs):
        input = v.output
        #input = utils.PrintShapeOp(input, 'pool')

        shp = input.shape
        upsample = T.zeros((shp[0], shp[1], shp[2] * self.pooling_size[0], shp[3] * self.pooling_size[1]),
                           dtype=input.dtype)
        output = T.set_subtensor(upsample[:, :, ::self.pooling_size[0], ::self.pooling_size[1]], input)

        print 'unpooling', self.input_height, self.input_width, self.pooling_size
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def get_output_height(self):
        return self.input_height * self.pooling_size[0]

    def get_output_width(self):
        return self.input_width * self.pooling_size[1]

    def __str__(self):
        d = [
            ('name', self.name),
            ('pooling_size', self.pooling_size),
            ('act_h', self.get_output_height()),
            ('act_w', self.get_output_width())]

        return 'UnPooling ' + utils.list_desc(d)

class MaxPoolingCudnn(Module):
    def __init__(self, input_height, input_width, pooling_size, pooling_stride, padding, name=''):
        super(MaxPoolingCudnn, self).__init__(name)
        self.params, self.opt_params, self.reg_params = [], [], []

        self.pooling_size = force_pair(pooling_size)
        self.pooling_stride = force_pair(pooling_stride)
        self.padding = force_pair(padding)
        self.input_height = input_height
        self.input_width = input_width

        if pooling_stride > pooling_size:
            raise NotImplementedError("MaxPooling only supports "
                                      "stride <= pool_size.")

    def apply(self, v, **kwargs):
        input = v.output
        #input = utils.PrintShapeOp(input, 'pool')
        if kwargs.get('use_cpu', False):
            output = downsample.max_pool_2d(input=input, ds=self.pooling_size, st=self.pooling_stride)
        else:
            output = dnn.dnn_pool(img=input, ws=self.pooling_size, stride=self.pooling_stride, pad=self.padding)

        print self.input_height, self.pooling_size, self.pooling_stride
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def get_output_height(self):
        return doit(self.input_height, self.padding[0], self.pooling_size[0], self.pooling_stride[0])

    def get_output_width(self):
        return doit(self.input_width, self.padding[1], self.pooling_size[1], self.pooling_stride[1])

    def __str__(self):
        d = [
            ('name', self.name),
            ('pooling_size', self.pooling_size),
            ('pooling_stride', self.pooling_stride),
            ('act_h', self.get_output_height()),
            ('act_w', self.get_output_width()),
            ('padding', self.padding)]

        return 'MaxPooling ' + utils.list_desc(d)


class Concatenate(Module):
    def __init__(self, axis, name=''):
        super(Concatenate, self).__init__(name)
        self.axis = axis
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        print 'getou_output_for conca', v
        print v
        nv = {}

        input_bunches = v
        input_tensors = map(lambda b: b.output, input_bunches)
        output = T.concatenate(input_tensors, self.axis)

        for bunch in input_bunches:
            nv.update(bunch)
        nv = Bunch(nv)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = []
        return 'Concatenate ' + utils.list_desc(d)


class Subtensor(Module):
    def __init__(self, index_var, name=''):
        super(Subtensor, self).__init__(name)
        self.index_var = index_var
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        input = v.output
        output = input[self.index_var]
        #input = utils.PrintShapeOp(input, 'pool')

        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = []
        return 'Subtensor ' + utils.list_desc(d)


class Dimshuffle(Module):
    def __init__(self, pattern, name=''):
        super(Dimshuffle, self).__init__(name)
        self.pattern = pattern
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v, **kwargs):
        input = v.output
        output = input.dimshuffle(self.pattern)
        #input = utils.PrintShapeOp(input, 'pool')

        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = []
        return 'Dimshuffle ' + utils.list_desc(d)

# TODO: gradient clipping
class LSTMStep(Module):
    def __init__(self, input_size, hidden_size, Tw_init, Tb_init, name='', forget_bias=None):

        super(LSTMStep, self).__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Tw_init = Tw_init
        self.Tb_init = Tb_init
        self.forget_bias = forget_bias

    def _allocate_params(self):
        self.Tw = self.create_shared('Tw', self.Tw_init, (self.input_size + self.hidden_size, 4 * self.hidden_size))
        self.Tb = self.create_shared('Tb', self.Tb_init, (4 * self.hidden_size,))

        if self.forget_bias is not None:
            Tb_value = self.Tb.get_value()
            Tb_value[self.hidden_size: 2 * self.hidden_size] = self.forget_bias
            self.Tb.set_value(Tb_value)

        all_params = [self.Tw, self.Tb]
        self.params, self.opt_params, self.reg_params = all_params, all_params, [self.Tw]


    def apply(self, v, **kwargs):
        """
        Required:

        v.h_tm1
        v.c_tm1
        v.x_t


        Result:
        v.h_t
        v.c_t
        """

        h_tm1, c_tm1, x_t = v.h_tm1, v.c_tm1, v.x_t

        layer_size = self.hidden_size
        print x_t.ndim, h_tm1.ndim
        M = T.dot(T.concatenate([h_tm1, x_t], axis=1), self.Tw) + self.Tb
        input_gate = sigmoid(M[:, 0:layer_size])
        forget_gate = sigmoid(M[:, layer_size: 2 * layer_size])
        output_gate = sigmoid(M[:, 2 * layer_size: 3 * layer_size])
        input_modulation_gate = tanh(M[:, 3 * layer_size: 4 * layer_size])

        c_t = forget_gate * c_tm1 + input_gate * input_modulation_gate
        h_t = output_gate * tanh(c_t)

        output = Bunch(c_t=c_t, h_t=h_t)
        return output

    def __str__(self):
        d = []
        return 'LSTMStep ' + utils.list_desc(d)

class LSTMStepGaussianNoise(Module):
    def __init__(self, input_size, hidden_size, Tw_init, Tb_init, name='', forget_bias=None):

        super(LSTMStepGaussianNoise, self).__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Tw_init = Tw_init
        self.Tb_init = Tb_init
        self.forget_bias = forget_bias


    def _allocate_params(self):
        self.Tw = self.create_shared('Tw', self.Tw_init, (self.input_size + self.hidden_size, 4 * self.hidden_size))
        self.Tb = self.create_shared('Tb', self.Tb_init, (4 * self.hidden_size,))

        if self.forget_bias is not None:
            Tb_value = self.Tb.get_value()
            Tb_value[self.hidden_size: 2 * self.hidden_size] = self.forget_bias
            self.Tb.set_value(Tb_value)

        all_params = [self.Tw, self.Tb]
        self.params, self.opt_params, self.reg_params = all_params, all_params, [self.Tw]


    def apply(self, v, **kwargs):
        """
        Required:

        v.h_tm1
        v.c_tm1
        v.x_t


        Result:
        v.h_t
        v.c_t
        """

        h_tm1, c_tm1, x_t, Tw_noise_t = v.h_tm1, v.c_tm1, v.x_t, v.Tw_noise_t

        layer_size = self.hidden_size
        print x_t.ndim, h_tm1.ndim
        M = T.dot(T.concatenate([h_tm1, x_t], axis=1), (self.Tw + Tw_noise_t)) + self.Tb
        input_gate = sigmoid(M[:, 0:layer_size])
        forget_gate = sigmoid(M[:, layer_size: 2 * layer_size])
        output_gate = sigmoid(M[:, 2 * layer_size: 3 * layer_size])
        input_modulation_gate = tanh(M[:, 3 * layer_size: 4 * layer_size])

        c_t = forget_gate * c_tm1 + input_gate * input_modulation_gate
        h_t = output_gate * tanh(c_t)

        output = Bunch(c_t=c_t, h_t=h_t)
        return output

    def __str__(self):
        d = []
        return 'LSTMStep ' + utils.list_desc(d)


# class LSTMGaussianNoise(Module):
#     def __init__(self, input_size, hidden_sizes, opt_init_state, save_state, gaussian_noise_sigma, name='', dropout_p_of_zero=None, max_mb_size=None):
#         super(LSTMGaussianNoise, self).__init__(name)
#
#         self.hidden_modules = []
#         self.hidden_sizes = hidden_sizes
#         self.dropout_p_of_zero = dropout_p_of_zero
#         self.save_state = save_state
#         self.gaussian_noise_sigma = gaussian_noise_sigma
#         last_size = input_size
#         self.c0s, self.h0s = [], []
#
#
#         self.params, self.opt_params, self.reg_params = [], [], []
#
#         for layer_idx, hidden_size in enumerate(hidden_sizes):
#             step_module = LSTMStepGaussianNoise(input_size=last_size, hidden_size=hidden_size,
#                                    Tw_init=Uniform(0.1), Tb_init=Constant(0.0), forget_bias=3,
#                                    name=name + '.' + ('Layer_%d' % layer_idx))
#             self.hidden_modules.append(step_module)
#
#
#             self.params += step_module.get_params()
#             self.opt_params += step_module.get_opt_params()
#             self.reg_params += step_module.get_reg_params()
#
#             if save_state:
#                 self.c0s.append(self.create_shared(shared_name=('c0_layer_%d' % layer_idx),
#                                               initializer=Constant(0.0),
#                                               shape=(max_mb_size, hidden_size)))
#
#                 self.h0s.append(self.create_shared(shared_name=('h0_layer_%d' % layer_idx),
#                                               initializer=Constant(0.0),
#                                               shape=(max_mb_size, hidden_size,)))
#             else:
#                 self.c0s.append(self.create_shared(shared_name=('c0_layer_%d' % layer_idx),
#                                               initializer=Constant(0.0),
#                                               shape=(1, hidden_size)))
#
#                 self.h0s.append(self.create_shared(shared_name=('h0_layer_%d' % layer_idx),
#                                               initializer=Constant(0.0),
#                                               shape=(1, hidden_size,)))
#
#             last_size = hidden_size
#
#         if opt_init_state:
#             self.params += self.c0s + self.h0s
#             self.opt_params += self.c0s + self.h0s
#
#         self.reg_params += [] #self.c0s + self.h0s
#         self.mb_size = max_mb_size
#
#     def reset_state(self):
#         for a in self.c0s:
#             shape = a.get_value().shape
#             a.set_value(np.zeros(shape, dtype=floatX))
#
#         for a in self.h0s:
#             shape = a.get_value().shape
#             a.set_value(np.zeros(shape, dtype=floatX))
#
#     def apply(self, v,**kwargs):
#         """
#         Data should be in T x MB x D format
#
#         Required:
#
#         v.output
#
#
#         Result:
#         v.output
#         """
#
#         input = v.output
#         mode = kwargs['mode']
#         # Previously we had the data in format MB x T x D, now we want T x MB x D to be able to scan over it
#
#         x = input
#
#         real_mb_size = T.cast(x.shape[1], 'int32')
#
#         def generate_next_t(c_tm1, h_tm1, x_t, Tw_noise_t, hidden_module):
#             v = hidden_module.apply(Bunch(c_tm1=c_tm1, h_tm1=h_tm1, x_t=x_t, Tw_noise_t=Tw_noise_t))
#             return [v.c_t, v.h_t]
#
#         new_c0, new_h0 = [], []
#
#         layer_idx = 0
#         for c0, h0, hidden_module in zip(self.c0s, self.h0s, self.hidden_modules):
#
#             # The general order of function parameters to fn is:
#             # sequences (if any), prior result(s) (if needed), non-sequences (if any)
#             if self.save_state:
#                 c0_ready = c0[0: real_mb_size]
#                 h0_ready = h0[0: real_mb_size]
#             else:
#                 c0_ready = T.repeat(c0, real_mb_size, axis=0),
#                 h0_ready = T.repeat(h0, real_mb_size, axis=0)
#
#             noise_shape = (x.shape[0],) + hidden_module.Tw.get_value().shape
#
#             if mode == 'train':
#                 Tw_noise = gaussian_noise(noise_shape, 0, self.gaussian_noise_sigma, layer_idx + 5)
#             elif mode == 'test':
#                 Tw_noise = T.zeros(noise_shape, dtype=floatX)
#             else:
#                 raise RuntimeError()
#
#             ([c_ts, h_ts], updates_scan) = theano.scan(
#                 fn=lambda x_t, Tw_noise_t, c_tm1, h_tm1: generate_next_t(c_tm1, h_tm1, x_t, Tw_noise_t, hidden_module),
#                 outputs_info=[c0_ready, h0_ready],
#                 sequences=[x, Tw_noise]
#             )
#
#             if self.save_state:
#                 new_c0.append(c_ts[-1])
#                 new_h0.append(h_ts[-1])
#
#             x = h_ts
#             if self.dropout_p_of_zero is not None:
#                 print 'Dropout of ', self.dropout_p_of_zero
#                 x = Dropout(self.dropout_p_of_zero).apply(Bunch(output=x), **kwargs).output
#             else:
#                 print 'No dropout'
#
#             layer_idx += 1
#
#
#         output = x
#         nv = vcopy(v)
#
#         if self.save_state:
#             if 'updates' not in nv:
#                 nv.updates = []
#
#             for a, b in zip(self.c0s, new_c0):
#                 nv.updates.append((a, T.set_subtensor(a[0: real_mb_size], b)))
#
#             for a, b in zip(self.h0s, new_h0):
#                 nv.updates.append((a, T.set_subtensor(a[0: real_mb_size], b)))
#
#         nv.update(output=output)
#         return nv
#
#     def __str__(self):
#         d = [('hidden_sizes', self.hidden_sizes)]
#         return 'LSTMGaussianNoise ' + utils.list_desc(d)



# WARNING! OLD ARCHITECTURE, add _allocate_params!!!, children
class LSTM(Module):
    def __init__(self, input_size, hidden_sizes, opt_init_state, save_state, name='', dropout_p_of_zero=None, max_mb_size=None):
        super(LSTM, self).__init__(name)

        self.hidden_modules = []
        self.hidden_sizes = hidden_sizes
        self.dropout_p_of_zero = dropout_p_of_zero
        self.save_state = save_state
        last_size = input_size
        self.c0s, self.h0s = [], []

        self.params, self.opt_params, self.reg_params = [], [], []

        for layer_idx, hidden_size in enumerate(hidden_sizes):
            step_module = LSTMStep(input_size=last_size, hidden_size=hidden_size,
                                   Tw_init=Uniform(0.1), Tb_init=Constant(0.0), forget_bias=3,
                                   name=name + '.' + ('Layer_%d' % layer_idx))
            self.hidden_modules.append(step_module)


            self.params += step_module.get_params()
            self.opt_params += step_module.get_opt_params()
            self.reg_params += step_module.get_reg_params()

            if save_state:
                self.c0s.append(self.create_shared(shared_name=('c0_layer_%d' % layer_idx),
                                              initializer=Constant(0.0),
                                              shape=(max_mb_size, hidden_size)))

                self.h0s.append(self.create_shared(shared_name=('h0_layer_%d' % layer_idx),
                                              initializer=Constant(0.0),
                                              shape=(max_mb_size, hidden_size,)))
            else:
                self.c0s.append(self.create_shared(shared_name=('c0_layer_%d' % layer_idx),
                                              initializer=Constant(0.0),
                                              shape=(1, hidden_size)))

                self.h0s.append(self.create_shared(shared_name=('h0_layer_%d' % layer_idx),
                                              initializer=Constant(0.0),
                                              shape=(1, hidden_size,)))

            last_size = hidden_size

        if opt_init_state:
            self.params += self.c0s + self.h0s
            self.opt_params += self.c0s + self.h0s

        self.reg_params += [] #self.c0s + self.h0s
        self.mb_size = max_mb_size

    def reset_state(self, idx):
        for a in self.c0s:
            a_val = a.get_value()
            a_val[idx] = 0
            a.set_value(a_val)

        for a in self.h0s:
            a_val = a.get_value()
            a_val[idx] = 0
            a.set_value(a_val)



    def apply(self, v, **kwargs):
        """
        Data should be in T x MB x D format

        Required:

        v.output


        Result:
        v.output
        """

        input = v.output

        # Previously we had the data in format MB x T x D, now we want T x MB x D to be able to scan over it

        x = input

        real_mb_size = T.cast(x.shape[1], 'int32')

        def generate_next_t(c_tm1, h_tm1, x_t, hidden_module):
            v = hidden_module.apply(Bunch(c_tm1=c_tm1, h_tm1=h_tm1, x_t=x_t))
            return [v.c_t, v.h_t]

        new_c0, new_h0 = [], []

        nv = vcopy(v)
        for c0, h0, hidden_module in zip(self.c0s, self.h0s, self.hidden_modules):

            # The general order of function parameters to fn is:
            # sequences (if any), prior result(s) (if needed), non-sequences (if any)
            if self.save_state:
                c0_ready = c0[0: real_mb_size]
                h0_ready = h0[0: real_mb_size]
            else:

                c0_ready = T.repeat(c0, real_mb_size, axis=0)
                h0_ready = T.repeat(h0, real_mb_size, axis=0)


            ([c_ts, h_ts], updates_scan) = theano.scan(
                fn=lambda x_t, c_tm1, h_tm1: generate_next_t(c_tm1, h_tm1, x_t, hidden_module),
                outputs_info=[c0_ready, h0_ready],
                sequences=[x]
            )

            if self.save_state:
                new_c0.append(c_ts[-1])
                new_h0.append(h_ts[-1])

            x = h_ts
            if self.dropout_p_of_zero is not None:
                print 'Dropout of ', self.dropout_p_of_zero
                x = Dropout(self.dropout_p_of_zero).apply(Bunch(output=x), **kwargs).output
            else:
                print 'No dropout'

        output = x

        if self.save_state:
            if 'updates' not in nv:
                nv.updates = []

            for a, b in zip(self.c0s, new_c0):
                nv.updates.append((a, T.set_subtensor(a[0: real_mb_size], b)))

            for a, b in zip(self.h0s, new_h0):
                nv.updates.append((a, T.set_subtensor(a[0: real_mb_size], b)))

        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = [('hidden_sizes', self.hidden_sizes)]
        return 'LSTM ' + utils.list_desc(d)


class RecurrentApply(Module):
    def __init__(self, module, name=''):
        super(RecurrentApply, self).__init__(name)
        self.module = module

    def get_params(self):
        return self.module.get_params()

    def get_opt_params(self):
        return self.module.get_opt_params()

    def get_reg_params(self):
        return self.module.get_reg_params()

    def apply(self, v, **kwargs):
        """
        v.output format is T x MB x D
        """

        input = v.output

        def generate_t(x_t):
            """
            :param x_t:
            :return: y_t
            """
            print type(self.module)
            return [self.module.apply(Bunch(output=x_t), **kwargs).output]

        print 'chuj', input.type, type(input)
        # ATTENTION!!!! updates_scan

        (y_ts, updates_scan) = theano.scan(
                fn=lambda x_t: generate_t(x_t),
                outputs_info=[None],
                sequences=[input]
            )

        output = y_ts
        nv = vcopy(v)
        nv.update(output=output)
        return self.post_apply(nv, **kwargs)

    def __str__(self):
        d = []
        return 'RecurrentApply ' + utils.list_desc(d)



