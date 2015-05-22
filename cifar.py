# TODO:
#   * rozne warianty SGD
#   * layer array (addLayer function)
#   * print layer params (W.get_value())
#   * print layer params in cudaconvnet
#   * check different momentum definintion

# wnioski:
#  * cuda convnet nie dzieli przez 255, dlatego ma tak absurdalnie niskie initW w pierwszej warstwie

import os
import sys
import time

import numpy
import theano
import theano.tensor as T
from theano.sandbox.cuda import dnn
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
#from matplotlib import pyplot as plt

from pylearn2.datasets import cifar10
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import pylearn2
import sys

################################################################

floatX = theano.config.floatX
uint8 = 'uint8'
int32 = 'int32'

def npcast(value, dtype=floatX):
  return numpy.asarray(value, dtype=dtype)

########### Plotting #########################

def simple_plot_to_file(y, filename, ylim=None):
  fig, axes = plt.subplots(nrows=1, ncols=1)
  axes.plot(y)
  if ylim is not None:
    axes.set_ylim(ylim)
  plt.savefig(filename)

def plot_image_to_file(img, filepath, interpolation='none'):
  plt.imshow(img, cmap='gray', interpolation=interpolation)
  plt.savefig(filepath)


def plot_image(img, cmap='gray', interpolation='none'):
  plt.imshow(img, cmap=cmap, interpolation=interpolation)
  plt.show()

########### Shared dataset #########################

def shared_dataset(data_xy, borrow=True):
  """ Function that loads the dataset into shared variables

  The reason we store our dataset in shared variables is to allow
  Theano to copy it into the GPU memory (when code is run on GPU).
  Since copying data into the GPU is slow, copying a minibatch everytime
  is needed (the default behaviour if the data is not in a shared
  variable) would lead to a large decrease in performance.
  """
  data_x, data_y = data_xy
  shared_x = theano.shared(numpy.asarray(data_x,
                                         dtype=theano.config.floatX),
                           borrow=borrow)
  #TODO hack
  shared_y = theano.shared(numpy.asarray(data_y[:, 0],
                                         dtype=theano.config.floatX),
                           borrow=borrow)
  # When storing data on the GPU it has to be stored as floats
  # therefore we will store the labels as ``floatX`` as well
  # (``shared_y`` does exactly that). But during our computations
  # we need them as ints (we use labels as index, and if they are
  # floats it doesn't make sense) therefore instead of returning
  # ``shared_y`` we will have to cast it to int. This little hack
  # lets ous get around this issue
  return shared_x, T.cast(shared_y, 'int32')


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def hasz(x, n):
    rng = numpy.random.RandomState(1)
    xx = numpy.asarray(rng.normal(0,1.0, n * 3072))
    return (x.flatten() * xx).mean()

def load_data(preprocess='data_mean'):
    #CIFAR10
    train = cifar10.CIFAR10('train')
    test = cifar10.CIFAR10('test')

    #print 'hasz train ', hasz(train.X, 50000)
    #print 'hasz test ', hasz(test.X, 10000)

    a = train.X[0]
#    a /= 255
    print a.shape
    a = a.reshape((3, 32, 32))
    b = numpy.zeros((32, 32, 3))
    b[:, :, 0] = a[0, :, :]
    b[:, :, 1] = a[1, :, :]
    b[:, :, 2] = a[2, :, :]
    print b.shape
    print b
#    plot_image(b[:, :, :], cmap=None)

    dict = unpickle('../data/batches.meta')
    data_mean = dict['data_mean'].reshape(3072)
    print data_mean
    print train.X.mean(axis=0)

    # TODO: more preprocessing?
    if (preprocess=='data_mean'):
        train.X -= data_mean
        test.X -= data_mean

    if (preprocess=='toronto_prepro'):
        test.X -= train.X.mean(axis=0) #order matters
        train.X -= train.X.mean(axis=0)

    # train.X /= 255.0
    print type(train.X), type(train.y)
    print train.X.shape
    print train.y.shape

    #test.X /= 255.0
    print type(test.X), type(test.y)
    print test.X.shape
    print test.y.shape

    train_set_x_val = train.X[0:40000]
    train_set_y_val = train.y[0:40000]

    valid_set_x_val = train.X[40000:50000]
    valid_set_y_val = train.y[40000:50000]

    return (shared_dataset((train_set_x_val, train_set_y_val)), shared_dataset((valid_set_x_val, valid_set_y_val)), shared_dataset((test.X, test.y)))

def relu(x):
    return T.switch(x < 0, 0, x)

def ident(x):
    return x

class SoftMaxLayer(object):
    def __init__(self, input):
        self.output = T.nnet.softmax(input)
        self.params = []

class FullyConnectedLayer(object):
    def __init__(self, rng, input, initW, n_in, n_out, epsW, momW, wc, epsB, momB, activation=ident):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        W_values = numpy.asarray(
            #rng.uniform(
            #    low=-numpy.sqrt(6. / (n_in + n_out)),
            #    high=numpy.sqrt(6. / (n_in + n_out)),
            #    size=(n_in, n_out)
            #),
            rng.normal(0, initW, size=(n_in, n_out)),
            dtype=theano.config.floatX
        )

        W = theano.shared(value=W_values, name='W', borrow=True)

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        self.output = activation(T.dot(input, self.W) + self.b)

        # parameters of the model
        self.params = [(self.W, epsW, momW, wc), (self.b, epsB, momB, 0)]

class LocallyConnectedLayer(object):
    def calc_size(self, image_size, pad, filter_size, stride):
        return (image_size + pad - filter_size) // stride + 1
    def __init__(self, rng, input, filter_shape, image_shape, stride, pad, initW, activation=ident):
        """
        :param filter_shape: (number of filters, num input feature maps, filter height, filter width)
        :param image_shape: (batch size, num input feature maps, image height, image width)
        """
        assert image_shape[1] == filter_shape[1]

        size = (self.calc_size(image_shape[2], pad[0], filter_shape[2], stride[0]),
                self.calc_size(image_shape[3], pad[1], filter_shape[3], stride[1]))
        size = size + filter_shape
        print 'size=', size
        self.W = theano.shared(
            numpy.asarray(
                rng.normal(0.0, initW, size=size),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        Transformer = pylearn2.linear.local_c01b.Local(filters = self.W,
                            image_shape = image_shape[2:4],
                            input_axes = ('b','c',0,1),
                            output_axes = ('b','c',0,1),
                            kernel_stride = stride, pad=pad)

        contiguous_input = gpu_contiguous(input)
        self.output = activation(Transformer.lmul(contiguous_input) + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]



class ConvLayer(object):
    def __init__(self, rng, input, filter_shape, image_shape, initW, border_mode='valid', activation=ident):
        """
        Allocate a ConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        # fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        #fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
        # initialize weights with random weights
        # W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        #initW=0.01
        print "initW ", initW
        self.W = theano.shared(
            numpy.asarray(
                #rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                rng.normal(0.0, initW, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )


        conv_out = dnn.dnn_conv(input, self.W, border_mode=border_mode)

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = theano.shared(value=numpy.zeros(filter_shape[0], dtype=theano.config.floatX), borrow=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [(self.W,0.001,0.9,0.004), (self.b,0.002,0.9,0)]
    def printParams(self):
        print self.W, self.b

class PoolLayer(object):
    def __init__(self, rng, input, poolsize, stride, mode, pad=(0, 0), activation=ident):
        pooled_out = dnn.dnn_pool(img=input, ws=poolsize, stride=stride, mode=mode, pad=pad)
        self.output = activation(pooled_out)
        self.params = []


def gradient_updates_momentum(cost, params, scale=1.0):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of 4-tuples (theano.tensor.var.TensorVariable, learning_rate, momentum, wc)
            Parameters to compute gradient against

    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for (param, learning_rate, momentum, wc) in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        print 'learning_rate=', learning_rate, ' momentum=', momentum, ' wc=', wc
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)

        newupdate = momentum * param_update + T.grad(cost, param) + wc * param
        updates.append((param, param - scale * learning_rate * newupdate))
        updates.append((param_update, newupdate))
    return updates

def errors(y_pred, y):
    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(y_pred, y))
    else:
        raise NotImplementedError()

def evaluate(n_epochs=2000, nkerns=[32, 32, 64], batch_size=128):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(25512)

    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    layer0_input = x.reshape((batch_size, 3, 32, 32))

    # conv: (32,32) -> (32,32)
    layer0conv = ConvLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        border_mode=2,
        initW = 0.0001,
        activation=ident
    )

    layer0conv.printParams()

    # pool: (32,32) -> (16,16)
    layer0pool = PoolLayer(
        rng,
        input=layer0conv.output,
        poolsize=(3,3),
        stride=(2,2),
        mode='max',
        pad=(1, 1),
        activation=relu
    )

    # conv (16,16) -> (16,16)
    layer1conv = ConvLayer(
        rng,
        input=layer0pool.output,
        image_shape=(batch_size, nkerns[0], 16, 16),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        border_mode=2,
        activation=relu,
        initW = 0.01
    )

    # pool (16,16) -> (8,8)
    layer1pool = PoolLayer(
        rng,
        input=layer1conv.output,
        poolsize=(3, 3),
        stride=(2, 2),
       # mode='max'
        mode='average',
        pad=(1, 1)
    )

    # (8,8) -> (8,8)
    layer2conv = ConvLayer(
        rng,
        input=layer1pool.output,
        image_shape=(batch_size, nkerns[1], 8, 8),
        filter_shape=(nkerns[2], nkerns[1], 5, 5),
        border_mode=2,
        activation=relu,
        initW = 0.01
    )

    # (8,8) -> (4,4)
    layer2pool = PoolLayer(
        rng,
        input=layer2conv.output,
        poolsize=(3, 3),
        stride=(2, 2),
     #   mode='max'
        mode='average',
        pad=(1, 1)
    )

    #newlayer = LocallyConnectedLayer(rng, input=layer2pool.output,
    #                                 filter_shape=(nkerns[2], nkerns[2], 3, 3),
    #                                 image_shape=(batch_size, nkerns[2], 4, 4),
    #                                 pad=(1,1), stride=(1,1), initW=0.1, activation=ident)

    # the FullyConnectedLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 5 * 5),
    # or (500, 50 * 5 * 5) = (500, 1250) with the default values.
    layer3_input = layer2pool.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    #layer3 = FullyConnectedLayer(rng=rng, input=layer3_input, initW=0.1, n_in=nkerns[2] * 4 * 4, n_out=64, epsW=0.001, momW=0.9, wc=0.03, epsB=0.002, momB=0.9, activation=relu)
    layer3 = FullyConnectedLayer(rng=rng, input=layer3_input, initW=0.01, n_in=nkerns[2] * 4 * 4, n_out=10, epsW=0.001, momW=0.9, wc=1, epsB=0.002, momB=0.9, activation=ident)

    #layer4 = FullyConnectedLayer(rng=rng, input=layer3.output, initW=0.1, n_in=64, n_out=10, epsW=0.001, momW=0.9, wc=0.03, epsB=0.002, momB=0.9, activation=ident)

    layer5 = SoftMaxLayer(input=layer3.output)
    #layer5 = SoftMaxLayer(input=layer4.output)

    # the cost we minimize during training is the NLL of the model
    cost = -T.mean(T.log(layer5.output)[T.arange(y.shape[0]), y])

    y_pred = T.argmax(layer5.output, axis=1)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        errors(y_pred, y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        errors(y_pred, y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    #params = layer5.params + layer4.params + layer3.params + layer2pool.params + layer2conv.params + layer1pool.params + layer1conv.params + layer0pool.params + layer0conv.params
    params = layer5.params + layer3.params + layer2pool.params + layer2conv.params + layer1pool.params + layer1conv.params + layer0pool.params + layer0conv.params

    # create a list of gradients for all model parameters
    #grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    #updates = [
    #    (param_i, param_i - learning_rate * grad_i)
    #    for param_i, grad_i in zip(params, grads)
    #]

    train_model = theano.function(
        [index],
        cost,
    #    updates=updates,
        updates = gradient_updates_momentum(cost, params),
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    train_model2 = theano.function(
        [index],
        cost,
        #    updates=updates,
        updates = gradient_updates_momentum(cost, params, 0.1),
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    train_model3 = theano.function(
        [index],
        cost,
        #    updates=updates,
        updates = gradient_updates_momentum(cost, params, 0.01),
        givens = {
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch_start_time = time.clock()
        epoch = epoch + 1
        costs = []
        times = []
        last_time = epoch_start_time
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            #cost_ij = train_model(minibatch_index)

            if epoch < 70:
                if iter % 100 == 0: print 'train1'
                cost_ij = train_model(minibatch_index)
            elif epoch < 85:
                if iter % 100 == 0: print 'train2'
                cost_ij = train_model2(minibatch_index)
            else:
                if iter % 100 == 0: print 'train3'
                cost_ij = train_model3(minibatch_index)
            costs = costs + [cost_ij]
            times = times + [time.clock() - last_time]
            last_time = time.clock()

            if (iter + 1) % validation_frequency == 0:
                train_end_time = time.clock()
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
        epoch_end_time = time.clock()
        print 'ave time ', numpy.mean(times)
        print 'train error ', numpy.mean(costs[0:78]), ' ', numpy.mean(costs[78:2*78]), ' ', numpy.mean(costs[2*78:3*78]), ' ', numpy.mean(costs[3*78:4*78])
        print >> sys.stderr, ('Epoch run for %.2f seconds' % ((epoch_end_time - epoch_start_time)))
        print >> sys.stderr, ('Training took %.2f seconds' % ((train_end_time - epoch_start_time)))
        sys.stdout.flush()

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    evaluate()
