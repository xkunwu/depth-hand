#from multiprocessing import Pool as ThreadPool
from multiprocessing import Pool as ThreadPool
from functools import partial
import numpy as np
import h5py

lines = ['line {}'.format(l) for l in np.arange(10)]
num_line = len(lines)

numbers = np.arange(10)

array = np.empty(10)

def write_txt(args, h5file):
    li = args[0]
    line = args[1]
    #h5file['index'][li] = line
    array[li] = line

with h5py.File('multip', 'w') as h5file:
    h5file.create_dataset(
        'index',
        (num_line, 1),
        compression='lzf',
        dtype='i4'
    )

    thread_pool = ThreadPool()
    thread_pool.map(partial(write_txt, h5file=h5file), zip(range(num_line), numbers))
    thread_pool.close()
    thread_pool.join()
    
    h5file['index']

with h5py.File('multip', 'r') as h5file:
    print(h5file['index'])
