import tensorflow as tf
import numpy as np
from ml_lib.TensorLib.helpers import desc_variable
from ml_lib.ml_utils import mkdir_p


class TfModel(object):
    def save_variables(self, epoch_idx=None, filename=None, global_step=None, sess=None):
        import os
        if epoch_idx is not None:
            filepath = os.path.join(self.tf_saver_dir_path, 'model' + '_epoch_' + str(epoch_idx) + '.ckpt')
        elif filename is not None:
            filepath = os.path.join(self.tf_saver_dir_path, filename)
        else:
            raise RuntimeError()
        print 'saving mode to ', filepath

        return self.save_variables_(self.tf_saver, filepath, global_step, sess)

    def save_variables_(self, tf_saver, filepath, global_step=None, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        save_path = tf_saver.save(sess, filepath, global_step=global_step)
        print('Model saved in file: %s, filepath was %s' % (save_path, filepath))
        return save_path

    def restore_variables(self, filepath, sess=None):
        return self.restore_variables_(self.tf_saver, filepath, sess)

    def restore_variables_(self, tf_saver, filepath, sess=None):
        print 'Restore variables', filepath
        if filepath is None:
            raise RuntimeError('filepath should not be None')
        if sess is None:
            sess = tf.get_default_session()
        tf_saver.restore(sess, filepath)
        print('Model restored from {filepath}'.format(filepath=filepath))

    def create_variables_saver(self, tf_saver_dir_path):
        try:
            self.tf_saver = tf.train.Saver(max_to_keep=100, var_list=self.variables)
            self.tf_saver_dir_path = tf_saver_dir_path
            print 'tf_saver_dir_path = ', self.tf_saver_dir_path
        except ValueError as e:
            import traceback
            print traceback.format_exc()

    def variables_desc(self):
        all_trainable_variables = self.trainable_variables
        res = []
        for var in all_trainable_variables:
            s = 'Variable {name}, shape = {shape}, nof_params = {nof_params}'.format(
                name=var.name,
                shape=var.get_shape(),
                nof_params=np.prod(map(int, var.get_shape()))
            )
            res.append(s)
        return '\n'.join(res)

    def get_nof_params(self):
        all_trainable_variables = self.trainable_variables
        s = 0
        for var in all_trainable_variables:
            s += np.prod(map(int, var.get_shape()))
        return s

    def initialize(self):
        nof_params = self.get_nof_params()

        all_variables = self.variables
        all_trainable_variables = self.trainable_variables


        initialize_op = tf.initialize_variables(all_variables)
        initialize_op.run()
        print 'Variables initialized.'

        # if self.args.restore_checkpoint_path is not None:
        #     model.restore_variables(self.args.restore_checkpoint_path)

        print 'ALL VARIABLES:'
        print '\n'.join(map(desc_variable, all_variables))
        print ''

        print 'TRAINABLE VARIABLES:'
        print '\n'.join(map(desc_variable, all_trainable_variables))
        print ''

        print 'L2_REG_MULT:'
        print '\n'.join(
            map(lambda a: (desc_variable(a[0]) + ', l2_reg_mult = ' + str(a[1])), tf.get_collection('l2_reg')))
        print ''

        print 'NOF_PARAMS', nof_params
        return self

def create_my_model(variables, trainable_variables, train_fun=None, eval_fun=None, valid_fun=None):
    class MyModel(TfModel):
        def __init__(self, variables_, trainable_variables_):
            self.variables = variables_
            self.trainable_variables = trainable_variables_

        def eval_fun(self, *args, **kwargs):
            if eval_fun is not None:
                return eval_fun(*args, **kwargs)
            else:
                raise NotImplementedError()

        def train_fun(self, *args, **kwargs):
            if train_fun is not None:
                return train_fun(*args, **kwargs)
            else:
                raise NotImplementedError()

        def valid_fun(self, *args, **kwargs):
            if valid_fun is not None:
                return valid_fun(*args, **kwargs)
            else:
                raise NotImplementedError()


    return MyModel(variables, trainable_variables)



