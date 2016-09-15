import numpy as np
import tensorflow as tf


class BaseProjectManager(object):
    def set_neptune_context(self, neptune_ctx):
        self.neptune_ctx_ = neptune_ctx

    def set_property(self, key, value):
        if hasattr(self, 'neptune_ctx_'):
            self.neptune_ctx_.job.properties[key] = value


    def variables_desc(self):
        all_trainable_variables = tf.trainable_variables()
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
        all_trainable_variables = tf.trainable_variables()
        s = 0
        for var in all_trainable_variables:
            s += np.prod(map(int, var.get_shape()))
        return s

    def check_dummy(self, v, v2, dummy):
        if not dummy:
            return v
        else:
            return v2

    # NOTE: Saving the whole graph is probably possible, but maybe not well documented
    # as of now. Revisit in the future.

    def save_variables(self, epoch_idx=None, filename=None, global_step=None, sess=None):
        import os
        if epoch_idx is not None:
            filepath = os.path.join(self.tf_saver_dir_path, 'model' + '_epoch_' + str(epoch_idx) + '.ckpt')
        elif filename is not None:
            filepath = os.path.join(self.tf_saver_dir_path, filename)

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

    def create_variable_savers(self):
        try:
            self.tf_saver = tf.train.Saver(max_to_keep=100)
            self.tf_saver_dir_path = self.saver.get_path('checkpoints/tf_saver/')
            print 'tf_saver_dir_path = ', self.tf_saver_dir_path
        except ValueError as e:
            import traceback
            print traceback.format_exc()

    def do_train_epoch(self, epoch_idx):
        raise NotImplementedError()

    def do_valid_epoch(self, epoch_idx):
        raise NotImplementedError()