import tensorflow as tf

OPTIMIZERS = {
   'momentum': lambda learning_rate_pl: tf.train.MomentumOptimizer(learning_rate_pl, momentum=0.9),
   'adam': lambda learning_rate_pl: tf.train.AdamOptimizer(learning_rate_pl),
   'rmsprop': lambda learning_rate_pl: tf.train.RMSPropOptimizer(learning_rate_pl),
   'adagrad': lambda learning_rate_pl: tf.train.AdagradOptimizer(learning_rate_pl),
   'adadelta': lambda learning_rate_pl: tf.train.AdadeltaOptimizer(learning_rate_pl)
}