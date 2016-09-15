import theano
import theano.tensor as T
from TheanoLib import utils
from TheanoLib.modules import Module, vcopy
from TheanoLib.utils import PrintValueOp


# class TransformerLayer(MergeLayer):
#     """
#     Spatial transformer layer
#     The layer applies an affine transformation on the input. The affine
#     transformation is parameterized with six learned parameters [1]_.
#     The output is interpolated with a bilinear transformation.
#     Parameters
#     ----------
#     incoming : a :class:`Layer` instance or a tuple
#         The layer feeding into this layer, or the expected input shape. The
#         output of this layer should be a 4D tensor, with shape
#         ``(batch_size, num_input_channels, input_rows, input_columns)``.
#     localization_network : a :class:`Layer` instance
#         The network that calculates the parameters of the affine
#         transformation. See the example for how to initialize to the identity
#         transform.
#     downsample_factor : float or iterable of float
#         A float or a 2-element tuple specifying the downsample factor for the
#         output image (in both spatial dimensions). A value of 1 will keep the
#         original size of the input. Values larger than 1 will downsample the
#         input. Values below 1 will upsample the input.
#     References
#     ----------
#     .. [1]  Spatial Transformer Networks
#             Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
#             Submitted on 5 Jun 2015
#     Examples
#     --------
#     Here we set up the layer to initially do the identity transform, similarly
#     to [1]_. Note that you will want to use a localization with linear output.
#     If the output from the localization networks is [t1, t2, t3, t4, t5, t6]
#     then t1 and t5 determines zoom, t2 and t4 determines skewness, and t3 and
#     t6 move the center position.
#     >>> import numpy as np
#     >>> import lasagne
#     >>> b = np.zeros((2, 3), dtype='float32')
#     >>> b[0, 0] = 1
#     >>> b[1, 1] = 1
#     >>> b = b.flatten()  # identity transform
#     >>> W = lasagne.init.Constant(0.0)
#     >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
#     >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=6, W=W, b=b,
#     ... nonlinearity=None)
#     >>> l_trans = lasagne.layers.TransformerLayer(l_in, l_loc)
#     """


class Transformer(Module):
    def __init__(self, name='', downsample_factor=1):
        super(Transformer, self).__init__(name)
        self.downsample_factor = (downsample_factor,) * 2
        self.params, self.opt_params, self.reg_params = [], [], []

    def apply(self, v_list, **kwargs):
        # conv shape should be MB x C x H x W
        # localisation shape should be MB x 6

        localisation_v = v_list[0]
        conv_v = v_list[1]

        output = _transform(localisation_v.output, conv_v.output, self.downsample_factor)

        output_v = vcopy(conv_v)
        output_v.update(vcopy(localisation_v))
        output_v.update(output=output)
        return self.post_apply(output_v, **kwargs)

    def __str__(self):
        d = [('name', self.name)]
        return 'Transformer ' + utils.list_desc(d)


    def get_output_shape_for(self, input_shapes):
        shape = input_shapes[0]
        factors = self.downsample_factor
        return (shape[:2] + tuple(None if s is None else int(s / f)
                                  for s, f in zip(shape[2:], factors)))



##########################
#    TRANSFORMER LAYERS  #
##########################

# copied from Lasagne

def _transform(theta, input, downsample_factor):
    num_batch, num_channels, height, width = input.shape
    theta = T.reshape(theta, (-1, 2, 3))

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = T.cast(height / downsample_factor[0], 'int64')
    out_width = T.cast(width / downsample_factor[1], 'int64')
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    T_g = T.dot(theta, grid)
    x_s = T_g[:, 0]
    y_s = T_g[:, 1]
    x_s_flat = x_s.flatten()
    y_s_flat = y_s.flatten()

    # dimshuffle input to  (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
    input_transformed = _interpolate(
        input_dim, x_s_flat, y_s_flat,
        out_height, out_width)

    output = T.reshape(
        input_transformed, (num_batch, out_height, out_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
    return output


def _interpolate(im, x, y, out_height, out_width):
    # *_f are floats
    num_batch, height, width, channels = im.shape
    height_f = T.cast(height, theano.config.floatX)
    width_f = T.cast(width, theano.config.floatX)

    # clip coordinates to [-1, 1]
    x = T.clip(x, -1, 1)
    y = T.clip(y, -1, 1)

    # scale coordinates from [-1, 1] to [0, width/height - 1]
    x = (x + 1) / 2 * (width_f - 1)
    y = (y + 1) / 2 * (height_f - 1)

    # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
    # we need those in floatX for interpolation and in int64 for indexing. for
    # indexing, we need to take care they do not extend past the image.
    x0_f = T.floor(x)
    y0_f = T.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1
    x0 = T.cast(x0_f, 'int64')
    y0 = T.cast(y0_f, 'int64')
    x1 = T.cast(T.minimum(x1_f, width_f - 1), 'int64')
    y1 = T.cast(T.minimum(y1_f, height_f - 1), 'int64')

    # The input is [num_batch, height, width, channels]. We do the lookup in
    # the flattened input, i.e [num_batch*height*width, channels]. We need
    # to offset all indices to match the flat version
    dim2 = width
    dim1 = width*height
    base = T.repeat(
        T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
    base_y0 = base + y0*dim2
    base_y1 = base + y1*dim2
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # use indices to lookup pixels for all samples
    im_flat = im.reshape((-1, channels))
    Ia = im_flat[idx_a]
    Ib = im_flat[idx_b]
    Ic = im_flat[idx_c]
    Id = im_flat[idx_d]

    # calculate interpolated values
    wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
    wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
    wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
    wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
    output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
    return output


def _linspace(start, stop, num):
    # Theano linspace. Behaves similar to np.linspace
    start = T.cast(start, theano.config.floatX)
    stop = T.cast(stop, theano.config.floatX)
    num = T.cast(num, theano.config.floatX)
    step = (stop-start)/(num-1)
    return T.arange(num, dtype=theano.config.floatX)*step+start


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(T.ones((height, 1)),
                _linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    y_t = T.dot(_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid