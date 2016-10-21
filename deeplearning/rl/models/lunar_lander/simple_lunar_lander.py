import tensorflow as tf

from ml_lib.TensorLib.create_functions_helpers import create_train_function, create_eval_function

from ml_lib.TensorLib.helpers import clip, apply_fc, create_apply_fc
from ml_lib.TensorLib.model import create_my_model



def create(conf):
    actions_n = conf['actions_n']
    name = conf['name']

    with tf.variable_scope(name):
        q_target = tf.placeholder(tf.float32, (None,), name='q_target')
        a_mask = tf.placeholder(tf.float32, (None, actions_n), name='target')
        state = tf.placeholder(tf.float32, (None, 8), name='state')
        res = state

        res = create_apply_fc(res, n_output=20, n_input=8, activation=tf.nn.relu, name='fc1',
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        res = create_apply_fc(res, n_output=20, n_input=20, activation=tf.nn.relu, name='fc2',
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        # res = create_apply_fc(res, n_output=16, n_input=16, activation=tf.nn.relu, name='fc3',
        #                       weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        res = create_apply_fc(res, n_output=actions_n, n_input=20, activation=None, name='fc4_actions', biases_initializer=None,
                              weights_initializer=tf.truncated_normal_initializer(stddev=0.1))
        q_s_a = res

        loss = tf.reduce_mean(tf.square(tf.reduce_sum(q_s_a * a_mask, reduction_indices=[1]) - q_target))
        mse = loss
        to_optimize = loss

        eval_out_dict = {'q': q_s_a}
        eval_func_spec = {'s': state}

        train_out_dict = {'q': q_s_a, 'loss': loss, 'mse': mse}
        train_func_spec = {'s': state, 'a_mask': a_mask, 'q_target': q_target}

        eval_fun = create_eval_function(eval_func_spec, eval_out_dict)
        train_fun = create_train_function(train_func_spec, to_optimize, train_out_dict, optimizer_creator=conf['optimizer_creator'])


        all_variables = tf.all_variables()
        my_variables = []
        my_trainable_variables = []

        for variable in all_variables:
            if variable.name.startswith(name + '/'):
                my_variables.append(variable)
                if variable in tf.trainable_variables():
                    my_trainable_variables.append(variable)

        # my_variables = tf.all_variables()
        # my_trainable_variables = tf.trainable_variables()
        return create_my_model(my_variables, my_trainable_variables, eval_fun=eval_fun, train_fun=train_fun)


