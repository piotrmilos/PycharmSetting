import copy
from collections import OrderedDict

import tensorflow as tf
from bunch import Bunch


def walk(act, fun):
        if isinstance(act, Bunch):
            return Bunch({k: walk(v, fun) for (k, v) in act.iteritems()})
        elif isinstance(act, dict):
            return {k: walk(v, fun) for (k, v) in act.iteritems()}
        elif isinstance(act, OrderedDict):
            return {k: walk(v, fun) for (k, v) in act.iteritems()}
        elif isinstance(act, list):
            return map(lambda a: walk(a, fun), act)
        elif isinstance(act, str):
            return act
        else:
            return fun(act)


def flatten_walk(w):
    outputs_flat = []

    def f(act):
        outputs_flat.append(act)
        return len(outputs_flat) - 1

    annotated = walk(w, f)
    return annotated, outputs_flat


def unflatten_walk(res_flat, annotated):
    def g(act):
        if isinstance(act, int):
            return res_flat[act]
        else:
            raise RuntimeError('Got ' + type(act) + ' should be int')

    ret_structured = walk(annotated, g)
    return ret_structured


def create_eval_function(func_spec, to_comp):
    def eval_function(**kwargs):
        feed_dict = {}
        for k, v in kwargs.iteritems():
            if k not in func_spec:
                raise RuntimeError('Error, unknown argument {key}!, func_spec = {func_spec}'.format(
                    key=k, func_spec=str(func_spec)
                ))
            feed_dict[func_spec[k]] = v

        sess = tf.get_default_session()
        annotated, to_comp_flatten = flatten_walk(to_comp)

        result_flat = sess.run(to_comp_flatten, feed_dict=feed_dict)
        result_structured = unflatten_walk(result_flat, annotated)

        return result_structured

    return eval_function

# NOTE(maciek): Same as eval, and part of train, refactor this
def create_valid_function(func_spec, to_comp):
    def valid_function(**kwargs):
        feed_dict = {}
        for k, v in kwargs.iteritems():
            if k not in func_spec:
                raise RuntimeError('Error, unknown argument {key}!, func_spec = {func_spec}'.format(
                    key=k, func_spec=str(func_spec)
                ))
            feed_dict[func_spec[k]] = v

        sess = tf.get_default_session()
        annotated, to_comp_flatten = flatten_walk(to_comp)

        result_flat = sess.run(to_comp_flatten, feed_dict=feed_dict)
        result_structured = unflatten_walk(result_flat, annotated)

        return result_structured

    return valid_function


def create_train_function(func_spec, to_optmize, to_comp, optimizer_creator, verify_finite=None):
    """

    :param func_spec: dict: name -> tensor
    :param to_optmize:
    :param to_comp:
    :param verify_finite:
    :return:
    """
    # TODO(maciek): apply to other create*

    to_comp = copy.copy(to_comp)
    learning_rate_pl = tf.placeholder(tf.float32)
    optimizer = optimizer_creator(learning_rate_pl)
    #momentum = 0.9
    #optimizer = tf.train.MomentumOptimizer(learning_rate_pl, momentum=momentum)
    grads_and_vars = optimizer.compute_gradients(to_optmize, tf.trainable_variables())

    if verify_finite is not None:
        print 'Will be verifying gradients for finiteness.'
        grads_and_vars = map(lambda (grad, var): (verify_finite(grad), var),
                             grads_and_vars)

    apply_gradients_op = optimizer.apply_gradients(grads_and_vars)

    print 'will create function with spec'
    print func_spec

    def train_function(learning_rate=None, **kwargs):
        if learning_rate is None:
            raise RuntimeError('learning_rate params should not be none')
        #print 'func_spec', func_spec
        #print 'kwargs', kwargs

        feed_dict = {learning_rate_pl: learning_rate}
        for k, v in kwargs.iteritems():
            if k not in func_spec:
                raise RuntimeError('Error, unknown argument {key}!, func_spec = {func_spec}'.format(
                    key=k, func_spec=str(func_spec)
                ))
            feed_dict[func_spec[k]] = v

        # print 'Running train_function', 'shapes'
        # print 'x_arr.shape', x_arr.shape
        # print 'y_corr_arr.shape', y_corr_arr.shape
        # print 'learning_rate_val', learning_rate

        sess = tf.get_default_session()
        to_comp['apply_gradients_op'] = apply_gradients_op

        annotated, to_comp_flatten = flatten_walk(to_comp)
        #for k, v in feed_dict.iteritems():
        #    print k, v.shape

        #print 'next is sess.run'
        #print 'to_comp_flatten', type(to_comp_flatten)
        result_flat = sess.run(to_comp_flatten, feed_dict)
        #print 'sess.run completed'
        result_structured = unflatten_walk(result_flat, annotated)
        return result_structured

    return train_function