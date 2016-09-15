from Queue import Empty
import multiprocessing as mp
from multiprocessing import Process
from multiprocessing.pool import ThreadPool
from setproctitle import setproctitle
import traceback

from bunch import Bunch
import numpy as np
import time

TIMEOUT_IN_SECONDS = 60


# This has to be picklable
class EndMarker(object):
    pass


# This has to be picklable
class ExceptionMarker(object):
    def __init__(self, traceback):
        self.traceback = traceback

    def get_traceback(self):
        return self.traceback


# class MinibatchOutputDirector(object):
#     def __init__(self, mb_size, data_shape, output_partial_batches=False):
#         self.mb_size = mb_size
#         self.output_partial_batches = output_partial_batches
#         self.data_shape = data_shape
#
#     def handle_begin(self):
#         self._start_new_mb()
#
#     def handle_result(self, res):
#         self.current_batch.append(res)
#
#         if 'x' in res:
#             #print 'chu', self.current_mb_size, self.mb_size
#             self.mb_x[self.current_mb_size] = res.x
#
#         if 'y' in res:
#             self.mb_y[self.current_mb_size] = res.y
#
#         self.current_mb_size += 1
#
#         if self.current_mb_size == self.mb_size:
#             res = self._get_res()
#             self._start_new_mb()
#             return res
#         else:
#             return None
#
#     def handle_end(self):
#         print 'handle_end'
#         if self.output_partial_batches:
#             print 'OK', self.current_mb_size
#             if len(self.current_batch) != 0:
#                 return self._get_res()
#             else:
#                 return None
#         else:
#             print 'none'
#             return None
#
#     def _get_res(self):
#         return Bunch(batch=self.current_batch,
#                         mb_x=self.mb_x,
#                         mb_y=self.mb_y)
#
#     def _start_new_mb(self):
#         self.current_mb_size = 0
#         self.current_batch = []
#         self.mb_x = np.zeros(shape=(self.mb_size,) + self.data_shape, dtype=ml_utils.floatX)
#         self.mb_y = np.zeros(shape=(self.mb_size,), dtype=ml_utils.int32)


class ProcessFunc(object):
    def __init__(self, process_func, *args, **kwargs):
        self.process_func = process_func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, elem):
        setproctitle('cnn_worker_thread')
        recipe_result = self.process_func(elem, *self.args, **self.kwargs)
        return recipe_result


#

# def get_iterator(process_func, elements_to_process, output_director, pool_size=4, buffer_size=20, chunk_size=100):
#
#     def add_to_queue_func(buffer_queue, item):
#         while buffer_queue.full():
#             #print 'buffer is full, sleep for a while.', buffer_queue.qsize()
#             time.sleep(5)
#         #print 'put to buffer_queue'
#         buffer_queue.put(item)
#
#
#     return BufferedProcessor(
#         MultiprocessingChunkProcessor(process_func, elements_to_process, output_director, chunk_size=chunk_size, pool_size=pool_size),
#         buffer_size=buffer_size,
#         add_to_queue_func=add_to_queue_func,
#         name='get_valid_iterator').get_iterator()


class PipelineElement(object):
    def __init__(self):
        self.source_it = None

    def set_source_iterator(self, source_it):
        self.source_it = source_it

    def get_iterator(self):
        raise RuntimeError()


class MinibatchOutputDirector2(PipelineElement):
    from ml_utils import floatX

    def __init__(self, mb_size, x_shape, y_shape, x_dtype=floatX, y_dtype=floatX, output_partial_batches=False):
        super(MinibatchOutputDirector2, self).__init__()
        self.mb_size = mb_size
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.x_dtype = x_dtype
        self.y_dtype = y_dtype
        self.output_partial_batches = output_partial_batches

    def get_iterator(self):
        self._start_new_mb()
        try:
            while True:
                res = self.source_it.next()
                r = self.handle_result(res)
                if r is not None:
                    yield r
        except StopIteration:
            res = self.handle_end()
            if res is not None:
                yield res

    def handle_result(self, res):
        self.current_batch.append(res)

        if 'x' in res:
            #print 'chu', self.current_mb_size, self.mb_size
            self.mb_x[self.current_mb_size] = res.x

        if 'y' in res:
            self.mb_y[self.current_mb_size] = res.y

        self.current_mb_size += 1

        if self.current_mb_size == self.mb_size:
            res = self._get_res()
            self._start_new_mb()
            return res
        else:
            return None

    def handle_end(self):
        print 'handle_end'
        if self.output_partial_batches:
            print 'OK', self.current_mb_size
            if len(self.current_batch) != 0:
                return self._get_res()
            else:
                return None
        else:
            print 'none'
            return None

    def _get_res(self):
        return Bunch(batch=self.current_batch,
                        mb_x=self.mb_x,
                        mb_y=self.mb_y)

    def _start_new_mb(self):
        self.current_mb_size = 0
        self.current_batch = []
        self.mb_x = np.zeros(shape=(self.mb_size,) + self.x_shape, dtype=self.x_dtype)
        self.mb_y = np.zeros(shape=(self.mb_size,) + self.y_shape, dtype=self.y_dtype)


class MultiprocessingChunkProcessor(PipelineElement):
    def __init__(self, callable, pool_size):
        super(MultiprocessingChunkProcessor, self).__init__()
        self.callable = callable
        self.pool_size = pool_size

    def get_iterator(self):
        pool = mp.Pool(self.pool_size)

        for chunk in self.source_it:
            res = pool.map(self.callable, chunk)
            yield res

        pool.close()
        pool.join()


class ThreadedChunkProcessor(PipelineElement):
    def __init__(self, callable, pool_size):
        super(ThreadedChunkProcessor, self).__init__()
        self.callable = callable
        self.pool_size = pool_size

    def get_iterator(self):
        pool = ThreadPool(self.pool_size)

        for chunk in self.source_it:
            res = pool.map(self.callable, chunk)
            yield res

        pool.close()
        pool.join()


class IterToQueue(PipelineElement):
    def __init__(self, queue):
        super(IterToQueue, self).__init__()
        self.queue = queue

    def get_iterator(self):
        try:
            print 'tak'
            for item in self.source_it:
                print 'done'
                while self.queue.full():
                     print 'buffer is full, sleep for a while.', self.queue.qsize()
                     time.sleep(5)

                self.queue.put(item)
                print 'put_done'
                yield 'put_ok'
            print 'putting end marker'
            self.queue.put(EndMarker())
        except Exception as e:
            self.queue.put(ExceptionMarker(traceback.format_exc()))


class Emitter(PipelineElement):
    def __init__(self, elems):
        super(Emitter, self).__init__()
        self.elems = elems

    def get_iterator(self):
        for elem in self.elems:
            print 'ok'
            yield elem


class UnChunker(PipelineElement):
    def __init__(self):
        super(UnChunker, self).__init__()

    def get_iterator(self):
        for chunk in self.source_it:
            for chunk_el in chunk:
                yield chunk_el


class Chunker(PipelineElement):
    def __init__(self, chunk_size):
        super(Chunker, self).__init__()
        self.chunk_size = chunk_size

    def get_iterator(self):
        res = []
        for a in self.source_it:
            res.append(a)
            if len(res) == self.chunk_size:
                yield res
                res = []
        if len(res):
            yield res

class QueueIter(PipelineElement):
    def __init__(self, queue, process_to_join=None, timeout=TIMEOUT_IN_SECONDS):
        super(QueueIter, self).__init__()
        self.queue = queue
        self.timeout = timeout
        self.process_to_join = process_to_join

    def get_iterator(self):
        while True:
            try:
                print 'will try get'
                v = self.queue.get(self.timeout)
                print 'get done'
            except Empty:
                print 'something is going wrong, could not get from queue'
                raise

            if isinstance(v, EndMarker):
                break

            if isinstance(v, ExceptionMarker):
                raise RuntimeError(v.get_traceback())
            else:
                yield v
        if self.process_to_join is not None:
            self.process_to_join.join()

def pipeline_to_iterator(pipeline):
    iter = pipeline[0].get_iterator()
    for el in pipeline[1:]:
        el.set_source_iterator(iter)
        iter = el.get_iterator()
    return iter

def execute_pipeline(pipeline):
    iter = pipeline_to_iterator(pipeline)
    print 'Pipeline execution started'
    for el in iter:
        print el
        pass
    print 'Pipeline execution completed'

def execute_async(func, args):
    process = Process(target=func, args=args)
    process.start()
    return process


# def create_old_iterator(process_func, elements_to_process, output_director, pool_size=4, buffer_size=20, chunk_size=100):
#     buffer = mp.Queue(maxsize=buffer_size)
#
#     pipeline = [Emitter(elements_to_process),
#                 Chunker(chunk_size),
#                 MultiprocessingChunkProcessor(process_func, pool_size),
#                 UnChunker(chunk_size),
#                 output_director,
#                 IterToQueue(buffer)
#                 ]
#
#     writer_process = execute_async(execute_pipeline, (pipeline,))
#
#     reader_pipeline = [QueueIter(buffer, writer_process)]
#     return pipeline_to_iterator(reader_pipeline)

def create_standard_iterator(process_func, elements_to_process, output_director, pool_size=4, buffer_size=20, chunk_size=30):
    buffer_size = 10
    buffer = mp.Queue(maxsize=buffer_size)
    buffer2 = mp.Queue(maxsize=3)
    chunk_size = 48

    pipeline1 = [Emitter(elements_to_process),
                Chunker(chunk_size),
                ThreadedChunkProcessor(process_func, pool_size=1),
                #MultiprocessingChunkProcessor(process_func, pool_size=30),
                IterToQueue(buffer2)]
    writer1_process = execute_async(execute_pipeline, (pipeline1,))

    # pipeline2 = [
    #             QueueIter(buffer2, writer1_process),
    #             #MultiprocessingChunkProcessor(process_func, pool_size=1),
    #             IterToQueue(buffer)
    #             #UnChunker(),
    #             #output_director,
    #             ]
    # writer2_process = execute_async(execute_pipeline, (pipeline2,))

    reader_pipeline = [QueueIter(buffer2, writer1_process)]

    #reader_pipeline = [QueueIter(buffer, writer2_process)]
    return pipeline_to_iterator(reader_pipeline)


