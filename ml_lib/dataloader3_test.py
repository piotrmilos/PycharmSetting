# from dataloader3 import get_iterator
import multiprocessing as mp
from multiprocessing import Process
from time import sleep
from dataloader3 import Chunker, Emitter, MultiprocessingChunkProcessor, IterToQueue, QueueIter

__author__ = 'maciek'
items = []
for a in xrange(20):
    items.append(a)


def process_func(x):
    sleep(2)
#    print 'process_func', x
    return x * 2


buffer = mp.Queue(maxsize=30)

pipeline = [Emitter(items),
            Chunker(2),
            MultiprocessingChunkProcessor(process_func, 2),
            IterToQueue(buffer)
            ]


writer_process = execute_async(execute_pipeline, (pipeline,))

reader_pipeline = [QueueIter(buffer)]
for el in pipeline_to_iterator(reader_pipeline):
    print el

writer_process.join()

