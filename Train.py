import tensorflow as tf
import Model

from datetime import datetime

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import Data2

batch_size = 8


def placeholder_inputs(batch_size):
    x_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                                Data2.IMAGE_SIZE, Data2.IMAGE_SIZE, 1))
    y_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return (x_placeholder, y_placeholder)

def fill_feed_dict(batch_size, data_set, x_pl, y_pl):
    x_feed, y_feed = data_set.next_batch(batch_size)
    feed_dict = {x_pl: x_feed, y_pl: y_feed}
    return feed_dict

#eval_correct is in fact Model.evaluation
def do_eval(batch_size, sess, eval_correct, x_pl, y_pl, data_set):
    correct_count = 0 #correct predictions nr
    steps_per_epoch = data_set.num_examples / batch_size
    num_examples = steps_per_epoch*batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(batch_size, data_set, x_pl, y_pl)
        correct_count += sess.run(eval_correct, feed_dict = feed_dict)
    precision = float(true_count) / num_examples
    print("for"+ str(data_set)+ " accuracy is %d / %d, "
                                "Precision @ 1: %0.04f")%(num_examples, correct_count, precision)

def run_training(epochs, batch_size, network):
    data_sets = input_data.read_data_sets("MNIST_data/", one_hot=True) #data_sets loading from mnist
    with tf.Graph().as_default():
        x_pl, y_pl = placeholder_inputs(batch_size)
        logits = network.run(x_pl) #HERE a particular network with given
        loss = Model.loss(logits, y_pl)
        train_op = Model.training(loss)
        eval_correct = Model.evaluation(logits, y_pl)

        summary_op = tf.merge_all_summaries()

        #initialize session
        saver = tf.train.Saver()
        sess.tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        summary_writer = tf.train.SummaryWriter(train_dir, sess.graph) #add train directory

        steps = train_set.num_examples / batch_size
        all_steps = epochs*steps
        monitor_rate = 4#this is somewhat global, batch_size is global,
                                    # rate = 4 means summary is 4 times per epoch

        for step in xrange(all_steps):
            start_time = datetime.now()
            print(start_time.strftime("%Y-%m-%d %H:%M:%S"))

            feed_dict = fill_feed_dict(batch_size, data_sets.train,
                                                    x_pl,
                                                    y_pl)
            _, loss_value = sess.run([train_op, loss], feed_dict = feed_dict)

            duration = start_time - datetime.now()

            #write summaries, , print evaluations
            for k in xrange(monitor_rate):
                if (step == k*steps/monitor_rate) or step == all_steps:
                    print('Epoch: %.3f, loss: %.2f '+str(duration)[:7])%(epoch+k*(steps/monitor_rate), loss_value, duration)
                    #updates events file
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    #save model
                    saver.save(sess, train_dir, global_step = step)
                    print('Training Data Eval:')
                    do_eval(batch_size, sess,
                                eval_correct,
                                x_pl,
                                y_pl,
                                data_sets.train)
                    # Evaluate against the validation set.
                    print('Validation Data Eval:')
                    do_eval(batch_size, sess,
                                 eval_correct,
                                 x_pl,
                                 y_pl,
                                 data_sets.validation)
                    # Evaluate against the test set.
                    print('Test Data Eval:')
                    do_eval(batch_size, sess,
                                 eval_correct,
                                 x_pl,
                                 y_pl,
                                 data_sets.test)

def main(_):

    layers_list = [Model.ConvLayer((batch_size, Data2.IMAGE_SIZE, Data2.IMAGE_SIZE, 1),
                                                            5, 10),
                               Model.PoolLayer(2),
                               Model.FCLayer(12*12*10 ,10)]
    net = Model.Network(layers_list)
    run_training(epochs=10, batch_size = batch_size, network = net)

if __name__ == '__main__':
    tf.app.run()

