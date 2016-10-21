from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPciInfo, \
    nvmlDeviceGetUtilizationRates
import sys
import time

import numpy as np

def get_utility(handle, what='gpu'):
    util = nvmlDeviceGetUtilizationRates(handle)
    if what == 'gpu':
        return util.gpu
    elif what == 'mem':
        return util.mem
    else:
        raise RuntimeError()

def main():
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    handles = []
    for i in range(0, deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        handles.append(handle)

    nof_tries = 40
    sleep_sec = 0.1
    max_utils = np.zeros((deviceCount,), 'float32')

    for try_idx in xrange(nof_tries):
        u = map(get_utility, handles)
        max_utils = np.maximum(max_utils, u)
        time.sleep(sleep_sec)
    #max_utils /= nof_tries
    #print max_utils
    gpu_id = np.argmin(max_utils)
    print 'gpu%d' % (gpu_id,)
    return


if __name__ == '__main__':
    sys.exit(main())
