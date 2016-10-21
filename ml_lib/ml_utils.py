import contextlib
import hashlib
import json
import os
import pickle
import random
import string
import time
import datetime
#from bokeh.io import cursession
import errno
from bunch import Bunch
import numpy as np
import numpy
from numpy.core.multiarray import ndarray
import image_utils

DEBUG = False

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4,
                    threshold=30000,
                    linewidth=150)


@contextlib.contextmanager
def timeit(text='', unit='ms'):
    timer = start_timer()
    try:
        yield
    finally:
        print text, 'Elapsed', elapsed_time_ms(timer), ' ', unit


def start_timer():
    return time.time()


def elapsed_time_secs(timer):
    return time.time() - timer


def elapsed_time_mins(timer):
    return (time.time() - timer) / 60


def elapsed_time_ms(timer):
    return (time.time() - timer) * 1000


def split_in_k(a, n):
    k, m = len(a) / n, len(a) % n
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))


def split_by_k(a, k):
    batch = []
    for v in a:
        batch.append(v)
        if len(batch) == k:
            yield batch
            batch = []
    if len(batch):
        yield batch


def repeat(examples, n_samples, mb_size):
    examples = trim(examples, mb_size)
    nexamples = []
    for a in xrange(len(examples) / mb_size):
        for i in xrange(n_samples):
            nexamples.extend(examples[a * mb_size: (a + 1) * mb_size])
    return nexamples



def unpack(a, *args):
    res = []
    for b in args:
        res.append(a[b])
    if len(res) == 1:
        return res[0]
    else:
        return tuple(res)


def mkdir_p(path):
    try:
        os.makedirs(path)
        return path
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            return path
        else:
            raise


def timestamp_str():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%H_%M_%S')

def timestamp_alt_str():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%y_%m_%d_%H_%M')


def REMOVE_ME(): # not this :P
    print 100 * 'Fuck! remove me'


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)


class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately

    def flush(self) :
        for f in self.files:
            f.flush()


def id_generator(n=10):
   return ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(n))


def small_line(n=10):
    print n * '='


def big_line(n=60):
    print n * '='


def epoch_header(epoch_idx):
    return '\n' + (50 * '-') + ' epoch_idx = ' + str(epoch_idx) + ' ' + (50 * '-') + '\n'


def mb_header(mb_idx):
    return '\n' + (20 * '-') + ' mb_idx = ' + str(mb_idx) + ' ' + (20 * '-') + '\n'

def fun_header(fun_name):
    return header(fun_name)

def header(s, n=50):
    r = (n * '-') + ' ' + s + ' ' + (n * '-')
    res = '\n'.join([len(r) * '-', r, len(r) * '-'])
    return res


def epoch_footer(epoch_idx):
    return (100 * '+')


def write_text_to_file(text, filepath):
    f = file(filepath, 'wb')
    f.write(text)
    f.close()


def read_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def read_lines(filepath):
    with open(filepath, 'r') as f:
        return f.readlines()


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def nphash(a):
    return hashlib.sha1(a).hexdigest()


def choice(t, n, numpy_rng):
    res = []
    for i in xrange(n):
        if numpy_rng is None:
            res.append(random.choice(t))
        else:
            res.append(t[numpy_rng.randint(0, len(t))])

    return res


def func_name():
    import traceback
    return traceback.extract_stack(None, 2)[0][2]

import random
def choose_one(t, rng=None):
    if rng is None:
        rng = random
    return t[rng.randint(0, len(t) - 1)]

def unique(l):
    return list(set(l))


def number_to_str(number, width=None):
    s = str(number)
    if width is not None:
        s = s.zfill(width)
    return s


class TimeSeries(object):
    def __getstate__(self):
        return {'t': self.t}

    def __savestate__(self, d):
        self.t = d['t']

    def __init__(self):
        self.t = []
        self.t_x = []

        self.add_observers = []

    def add(self, y, x=None):
        if x is None:
            x = (0 if len(self.t_x) == 0 else self.t_x[-1]) + 1

        self.t.append(y)
        self.t_x.append(x)

        for add_observer in self.add_observers:
            add_observer.notify_add(self, y, x)

    def add_add_observer(self, observer):
        self.add_observers.append(observer)

    def size(self):
        return len(self.t)

    def last_mean(self, n=None):
        if n is None:
            n = self.size()

        if n > self.size():
            raise RuntimeError()

        return np.mean(self.t[-n:])

    def get_items(self):
        return self.t

    def last_n(self, n):
        if n == -1:
            return self.t
        if n > self.size():
            raise RuntimeError()

        return self.t[-n:]

    def last_x(self):
        return self.t_x[-1]


# class BokehTimeseriesObserver(object):
#     def __init__(self, datasource, add_freq, fun=np.mean):
#         self.datasource = datasource
#         self.add_freq = add_freq
#         self.fun = fun
#
#     def notify_add(self, ts, y, x):
#         if ts.size() % self.add_freq == 0 and ts.size():
#             print 'Channel store_objects!'
#             self.datasource.data['x'].append(x)
#             self.datasource.data['y'].append(self.fun(ts.last_n(self.add_freq)))
#             cursession().store_objects(self.datasource)


class LogTimeseriesObserver(object):
    def __init__(self, name, add_freq, fun=np.mean):
        self.add_freq = add_freq
        self.fun = fun
        self.name = name

    def notify_add(self, ts, y, x):
        if ts.size() % self.add_freq == 0 and ts.size():
            xx = x
            yy = self.fun(ts.last_n(self.add_freq))
            print 'LogTimerseries {name} x={x}, y={y}'.format(name=self.name,
                                                              x=xx, y=yy)



DEEPLEARNING_HOME = os.environ.get('DEEPLEARNING_HOME', None)


def trim(examples, mb_size):
    l = len(examples)
    ll = l - l % mb_size
    examples = examples[0:ll]
    return examples


uint8 = 'uint8'
int32 = 'int32'
float32 = 'float32'
floatX = float32


def npcast(value, dtype=floatX):
    return np.asarray(value, dtype=dtype)

def get_part(l, part_idx, mod):
    return l[part_idx::mod]

def ArgparsePair(value):
    w = value.split('%')
    return int(w[0]), int(w[1])


# Rewrite to numpy
def get_top_k_accuracy(output, ans, k=5):
    sum = 0
    for idx in xrange(ans.shape[0]):
        w = np.argsort(output[idx])
        for a in xrange(k):
            if w[-a - 1] == ans[idx]:
                sum += 1
                break
    return sum / float(ans.shape[0])

def as_mb(a):
    return np.expand_dims(a, axis=0)

def binary_crossentropy(pred_dist, true_dist, eps=0):
    return -(true_dist * np.log(pred_dist) + (1 - true_dist) * np.log(1 - pred_dist))

def categorical_crossentropy(pred_dist, true_dist):

    if len(true_dist.shape) == 1 and true_dist.dtype in ['int32']:
        d = np.zeros_like(pred_dist)
        d[np.arange(0, pred_dist.shape[0]), true_dist] = 1.0
        true_dist = d
    elif len(true_dist.shape) == 1:
        true_dist = as_mb(true_dist)

    if len(pred_dist.shape) == 1:
        pred_dist = as_mb(pred_dist)

    loss = -(true_dist * np.log(pred_dist)).sum(axis=1)
    return loss


# TAKEN FROM deeplearning.net
def scale_to_unit_interval(ndar, eps=1e-8):
  """ Scales all values in the ndarray ndar to be between 0 and 1 """
  ndar = ndar.copy()
  ndar -= ndar.min()
  ndar *= 1.0 / (ndar.max() + eps)
  return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
  """
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.

  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).

  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.

  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image

  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)

  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats

  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not


  :returns: array suitable for viewing as an image.
  (See:`Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.

  """

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2

  # The expression below can be re-written in a more C style as
  # follows :
  #
  # out_shape = [0,0]
  # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
  #                tile_spacing[0]
  # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
  #                tile_spacing[1]
  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
      assert len(X) == 4
      # Create an output numpy ndarray to store the image
      if output_pixel_vals:
          out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
      else:
          out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

      #colors default to 0, alpha defaults to 1 (opaque)
      if output_pixel_vals:
          channel_defaults = [0, 0, 0, 255]
      else:
          channel_defaults = [0., 0., 0., 1.]

      for i in xrange(4):
          if X[i] is None:
              # if channel is None, fill it with zeros of the correct
              # dtype
              out_array[:, :, i] = numpy.zeros(out_shape,
                      dtype='uint8' if output_pixel_vals else out_array.dtype
                      ) + channel_defaults[i]
          else:
              # use a recurrent call to compute the channel and store it
              # in the output
              out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
      return out_array

  else:
      # if we are dealing with only one channel
      H, W = img_shape
      Hs, Ws = tile_spacing

      # generate a matrix to store the output
      out_array = numpy.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


      for tile_row in xrange(tile_shape[0]):
          for tile_col in xrange(tile_shape[1]):
              if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                  if scale_rows_to_unit_interval:
                      # if we should scale values to be between 0 and 1
                      # do this by calling the `scale_to_unit_interval`
                      # function
                      this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                  else:
                      this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                  # add the slice to the corresponding position in the
                  # output array
                  out_array[
                      tile_row * (H+Hs): tile_row * (H + Hs) + H,
                      tile_col * (W+Ws): tile_col * (W + Ws) + W
                      ] \
                      = this_img * (255 if output_pixel_vals else 1)
      return out_array


def show_filters(filters, filepath=None, kernel_size=(3, 3)):
            print 'Will draw filters'
            fshape = filters.shape
            print fshape
            res = np.zeros(shape=(fshape[0], 3, fshape[2] * fshape[3]))

            for a in xrange(fshape[0]):
                for b in xrange(0, 3):
                    res[a, b, :] = filters[a, b, ...].flatten('C')

            img = tile_raster_images((res[:, 0, ...], res[:, 1, ...], res[:, 2, ...], None),
                                                        img_shape=kernel_size,
                                                        tile_shape=(4, fshape[0] / 4),
                                                        tile_spacing=(1, 1))
            print img

            if filepath is None:
                image_utils.plot_image(img)
            else:
                image_utils.plot_image_to_file(img, filepath)

def show_filters_2(filters, filepath=None, kernel_size=(3, 3)):
    print 'Will draw filters'
    fshape = filters.shape
    print fshape
    res = np.zeros(shape=(fshape[0], 1, fshape[2] * fshape[3]))

    for a in xrange(fshape[0]):
        for b in xrange(0, 1):
            res[a, b, :] = filters[a, b, ...].flatten('C')

    img = tile_raster_images(res[:, 0, ...],
                                                img_shape=kernel_size,
                                                tile_shape=(4, fshape[0] / 4),
                                                tile_spacing=(1, 1))
    print img

    if filepath is None:
        image_utils.plot_image(img)
    else:
        image_utils.plot_image_to_file(img, filepath)

class NumpyAwareEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return str(obj)
            return json.JSONEncoder(self, obj)

def my_portable_hash(l):
    s = json.dumps(l, cls=NumpyAwareEncoder)
    print 'before hash', s
    return hashlib.sha224(s).hexdigest()[:20]


def save_obj(obj, path):
        print 'saving obj, ', path
        pickle.dump(obj, open(path, 'wb'))


def load_obj(path):
    print 'loading obj, ', path
    try:
        return pickle.load(open(path, 'rb'))
    except IOError:
        return None

def get_idx(idx):
    def f(a):
        return a[idx]
    return f

def create_get_attr(attr):
    def f(a):
        return getattr(a, attr)
    return f

def map_get_attr(attr, l):
    return map(lambda x: getattr(x, attr), l)

def map_get_attrs(attrs, l):
    return map(lambda x: tuple([getattr(x, attr) for attr in attrs]), l)

def map_get_key(key, l):
    return map(lambda x: x[key], l)

def print_join(s, l):
    print s.join(map(str, l))


def select_gpu():
    print 'Automatic gpu selection.'
    try:
        import pynvml
        pynvml.nvmlInit()
        numOfGPUs = int(pynvml.nvmlDeviceGetCount())
        res = []
        for i in range(0, numOfGPUs):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            try:
                memInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = memInfo.used / 2 ** 20
                mem_free = memInfo.free / 2 ** 20
            except pynvml.NVMLError as err:
                error = pynvml.handleError(err)
                mem_total = error
                mem_used = error
                mem_free = error
            print i, mem_free
            res.append((i, mem_free))

            # NOTE: You can also iterate processes running on this gpu.

            # procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            #
            # for p in procs:
            #     try:
            #         name = str(pynvml.nvmlSystemGetProcessName(p.pid))
            #         print name, p.usedGpuMemory / 2 ** 20
            #     except pynvml.NVMLError as err:
            #         if (err.value == pynvml.NVML_ERROR_NOT_FOUND):
            #             # probably went away
            #             continue
            #         else:
            #             name = handleError(err)
        gpu_id, free = sorted(res, key=lambda a: a[1])[-1]
        MINIMAL_FREE_MEMORY_MB = 1000

        if free < MINIMAL_FREE_MEMORY_MB:
            raise RuntimeError('No free gpu!')

        print 'Selecting device={id}, with {free} mbs of free memory'.format(id=gpu_id, free=free)
        return gpu_id

    except pynvml.NVMLError, err:
        print "Failed to initialize NVML: ", err
        print "Exiting..."
        os._exit(1)


RED = np.asarray([240, 0, 0])
GREEN = np.asarray([0, 240, 0])
BLUE = np.asarray([0, 0, 240])


def draw_rect(arr, coord1, coord2, color=RED, thickness=3):
    x1, y1 = coord1
    x2, y2 = coord2
    print 'draw rect x1={}, y1={}, x2={}, y2={}'.format(x1, y1, x2, y2)

    arr[y1:y2, x1: x1 + thickness] = color
    arr[y1:y2, x2: x2 + thickness] = color
    arr[y1: y1 + thickness, x1:x2] = color
    arr[y2: y2 + thickness, x1:x2] = color

def draw_rect_2(arr, coord1, coord2, color=RED, thickness=3):
    x1, y1 = coord1
    x2, y2 = coord2
    print 'draw rect x1={}, y1={}, x2={}, y2={}'.format(x1, y1, x2, y2)

    arr[x1:x2, y1: y1 + thickness] = color
    arr[x1:x2, y2: y2 + thickness] = color
    arr[x1: x1 + thickness, y1:y2] = color
    arr[x2: x2 + thickness, y1:y2] = color




