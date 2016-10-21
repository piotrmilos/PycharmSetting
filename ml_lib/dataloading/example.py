import traceback
import os
import sys
from time import sleep

import scipy
import signal
from bunch import Bunch

from dataloading.dataloader import ProcessFunc, SimpleOutputDirector, get_iterator
from ml_utils import start_timer, elapsed_time_ms


def prepare_batch(batch):
    batch_size = len(batch)
    print 'batch_size', batch_size

    for idx, item in enumerate(batch):
        print item.x.shape, item.recipe, item.info


def fetch_recipe(recipe, global_spec):
    try:
        path = recipe['path']
        timer = start_timer()
        img = scipy.misc.imread(path)
        print img.shape
        # simulate processing
        sleep(1)
        elapsed = elapsed_time_ms(timer)
        return Bunch(x=img, recipe=recipe, info={'elapsed': elapsed})
    except Exception as e:
        print traceback.format_exc()
        print recipe
        raise


def get_files(root_dir):
    return [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]


def create_recipes(l):
    def create_recipe(path):
        return {'path': path}
    return map(create_recipe, l)


def read_files_parallel(l):
    mb_size = 32
    recipes = create_recipes(l)
    global_spec = {}
    process_func = ProcessFunc(fetch_recipe, global_spec)
    output_director = SimpleOutputDirector(mb_size, output_partial_batches=True)
    iterator = get_iterator(process_func, recipes, output_director,
                            pool_size=10, buffer_size=40, chunk_size=mb_size * 3)
    for b in iterator:
        prepare_batch(b.batch)


def main():
    # This is important, we create lots of subprocesses, thanks to this we kill them with ctrl-c
    os.setpgrp()
    def install_sigterm_handler():
        def handleTerm(sign, no):
            print 'handleTerm', sign
            raise KeyboardInterrupt()

        signal.signal(signal.SIGTERM, handleTerm)
    install_sigterm_handler()


    root_dir = sys.argv[1]
    LIMIT = 100
    files = get_files(root_dir)[:LIMIT]
    files = map(lambda file: os.path.join(root_dir, file), files)

    timer = start_timer()
    read_files_parallel(files)
    elapsed = elapsed_time_ms(timer)
    print elapsed

    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)

if __name__ == '__main__':
    main()


