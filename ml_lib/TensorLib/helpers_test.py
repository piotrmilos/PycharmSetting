import unittest

import numpy as np
import tensorflow as tf
from tensorflow.python.framework.errors import InvalidArgumentError

from TensorLib.helpers import softmax3, add_at_position, shift_right, general_matmul, my_reshape


def softmax(w, t = 1.0):
    e = np.exp(w / t)
    dist = e / np.sum(e)
    return dist


class Test(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        ))

    def test_indexing_by_tensor_fail(self):
        # Does such things work in theano?
        with self.assertRaises(TypeError):
            a = tf.placeholder(tf.float32)
            b = tf.placeholder(tf.int32)
            res = a[b]

    def test_softmax3(self):
        a = tf.placeholder(tf.float32)
        res = softmax3(a)

        a_val = np.random.uniform(0, 1, (6, 7, 8))
        res_val = self.sess.run(res, feed_dict={a: a_val})
        for i in xrange(6):
            for j in xrange(7):
                print softmax(a_val[i, j, :]), res_val[i, j, :]
                self.assertTrue(np.allclose(softmax(a_val[i, j, :]), res_val[i, j, :]))

    def test_matmul_3d_fail(self):
        a = tf.placeholder(tf.int32)
        res = tf.matmul(a, a)
        a_val = np.zeros(shape=(6, 6, 6), dtype='int32')
        with self.assertRaises(InvalidArgumentError):
            self.sess.run(res, feed_dict={a: a_val})

    def test_add_at_position(self):
        a = tf.placeholder(tf.int32)
        res = add_at_position(a, (3, 4), 10)

        a_val = np.zeros(shape=(6, 6), dtype='int32')
        res_val = self.sess.run(res, feed_dict={a: a_val})
        for i in xrange(6):
            for j in xrange(6):
                if i == 3 and j == 4:
                    self.assertEqual(10, res_val[i, j])
                else:
                    self.assertEqual(0, res_val[i, j])

        print res_val

    def test_shift_right_dim1(self):
        a = tf.placeholder(tf.int32)
        k = 2
        res = shift_right(a, 1, 0, k)

        a_val = np.random.randint(low=0, high=100, size=(6, ), dtype='int32')
        res_val = self.sess.run(res, feed_dict={a: a_val})
        corr = np.concatenate([np.zeros(k, dtype='int32'), a_val[:-k]])
        print a_val
        print corr
        print res_val
        self.assertTrue(np.array_equal(corr, res_val))

    def test_shift_right_dim2(self):
        a = tf.placeholder(tf.int32)
        k1 = 2
        k2 = 3

        res = shift_right(a, 2, 0, k1)
        res = shift_right(res, 2, 1, k2)

        a_val = np.random.randint(low=0, high=100, size=(6, 10), dtype='int32')
        res_val = self.sess.run(res, feed_dict={a: a_val})

        corr = np.zeros((6, 10), dtype='int32')
        corr[k1:, k2:] = a_val[:-k1:, :-k2]
        print a_val
        print res_val
        print corr
        self.assertTrue(np.array_equal(corr, res_val))

    def test_general_matmul(self):
        a = tf.placeholder(tf.int32)
        b = tf.placeholder(tf.int32)
        res = general_matmul(a, 3, b, 3)

        a_val = np.random.randint(low=0, high=100, size=(2, 3, 4))
        b_val = np.random.randint(low=0, high=100, size=(4, 5, 6))
        res_val = self.sess.run(res, feed_dict={a: a_val, b: b_val})

        for a_i in xrange(2):
            for a_j in xrange(3):
                for b_i in xrange(5):
                    for b_j in xrange(6):
                        corr = np.sum(a_val[a_i, a_j, :] * b_val[:, b_i, b_j])
                        ans = res_val[a_i, a_j, b_i, b_j]
                        self.assertEqual(corr, ans)

        print res_val.shape


    def test_my_reshape(self):
        a = tf.placeholder(tf.int32)
        res = my_reshape(a, [6, 4])

        a_val = np.random.randint(low=0, high=100, size=(2, 3, 4))
        res_val = self.sess.run(res, feed_dict={a: a_val})
        print a_val
        print res_val
        self.fail('no assert done, write some')