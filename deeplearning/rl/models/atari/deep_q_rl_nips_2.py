import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, fully_connected, flatten

from ml_lib.TensorLib.create_functions_helpers import create_train_function, create_eval_function

from ml_lib.TensorLib.model import create_my_model

#def create_activation_stats(j)

def create(conf):
    actions_n = conf['actions_n']
    name = conf['name']
    print 'actions_n', actions_n

    with tf.variable_scope(name):
        q_target_pl = tf.placeholder(tf.float32, (None,), name='q_target')
        a_mask_pl = tf.placeholder(tf.float32, (None, actions_n), name='target')
        state_pl = tf.placeholder(tf.float32, (None, 4, 84, 84), name='state')
        res = state_pl / 255.0
        res = tf.transpose(res, perm=[0, 2, 3, 1])
        res = convolution2d(res, num_outputs=32, kernel_size=(8, 8), stride=(4, 4), activation_fn=tf.nn.relu,
                            weights_initializer=tf.random_normal_initializer(stddev=0.01),
                            biases_initializer=tf.constant_initializer(value=0.1),
                            )
        print 'conv1 shape', res.get_shape()
        conv1 = res

        res = convolution2d(res, num_outputs=64, kernel_size=(4, 4), stride=(2, 2), activation_fn=tf.nn.relu,
                            weights_initializer=tf.random_normal_initializer(stddev=0.01),
                            biases_initializer=tf.constant_initializer(value=0.1)
                            )
        print 'conv2 shape', res.get_shape()
        conv2 = res

        res = flatten(res)

        res = fully_connected(res, 512, activation_fn=tf.nn.relu,
                              weights_initializer=tf.random_normal_initializer(stddev=0.01),
                              biases_initializer=tf.constant_initializer(value=0.1)
                              )
        fc1 = res

        mean_non_zero_conv1_frac = tf.reduce_mean(tf.cast(conv1 > 0.0, tf.float32))
        mean_non_zero_conv2_frac = tf.reduce_mean(tf.cast(conv2 > 0.0, tf.float32))
        mean_non_zero_fc1_frac = tf.reduce_mean(tf.cast(fc1 > 0.0, tf.float32))

        mean_conv1 = tf.reduce_mean(tf.cast(conv1, tf.float32))
        mean_conv2 = tf.reduce_mean(tf.cast(conv2, tf.float32))
        mean_fc1 = tf.reduce_mean(tf.cast(fc1, tf.float32))


        res = fully_connected(res, actions_n, activation_fn=None,
                              weights_initializer=tf.random_normal_initializer(stddev=0.01),
                              biases_initializer=tf.constant_initializer(value=0.1)
        )

        q_s_a = res

        diff = tf.reduce_sum(q_s_a * a_mask_pl, reduction_indices=[1]) - q_target_pl
        loss = tf.reduce_mean(tf.square(diff))
        mae = tf.reduce_mean(tf.abs(diff))
        mse = loss

        to_optimize = loss

        eval_func_spec = {'s': state_pl}
        eval_out_dict = {'q': q_s_a}

        train_func_spec = {'s': state_pl, 'a_mask': a_mask_pl, 'q_target': q_target_pl}
        train_out_dict = {'q': q_s_a, 'loss': loss, 'mse': mse, 'mae': mae}
        train_out_dict.update({'conv1': conv1, 'conv2': conv2, 'fc1': fc1, 'q_s_a': q_s_a})

        train_out_dict.update({'mean_non_zero_fc1_frac': mean_non_zero_fc1_frac,
                               'mean_non_zero_conv1_frac': mean_non_zero_conv1_frac,
                               'mean_non_zero_conv2_frac': mean_non_zero_conv2_frac,
                               'mean_conv1': mean_conv1,
                               'mean_conv2': mean_conv2,
                               'mean_fc1': mean_fc1,
                               })

        eval_fun = create_eval_function(eval_func_spec, eval_out_dict)
        train_fun = create_train_function(train_func_spec, to_optimize, train_out_dict,
                                          optimizer_creator=conf['optimizer_creator'])


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


