# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html

import cupy as cp
import numpy as np
import time
import random
import nvtx
from cupyx.profiler import benchmark


def compute(array_1, array_2, a, b, c):
    def np_compute(array_1, array_2, a, b, c):
        return np.clip(array_1, 2, 10) * a + array_2 * b + c
    
    @cp.fuse(kernel_name='cp_compute')
    def cp_compute(array_1, array_2, a, b, c):
        return cp.clip(array_1, 2, 10) * a + array_2 * b + c

    # first time execution to reduce overhead
    cp_compute(np.random.randn(10), np.random.randn(10), np.int64(4), np.int64(3), np.int64(9))

    

    a_val_cp = cp.int64(4)
    b_val_cp = cp.int64(3)
    c_val_cp = cp.int64(9)
    a_val_np = np.int64(4)
    b_val_np = np.int64(3)
    c_val_np = np.int64(9)

    for array_length in range(10000, 200001, 10000):
    # for array_length in range(10000, 10001, 10000):
        initial_array = cp.random.randn(array_length)
        current_array = initial_array
        current_device = 'gpu'

        for op_length in range(10000, 100001, 10000):
        # for op_length in range(10000, 10001, 10000):
            for np_chance in range(0, 101, 10):
                operations = []
                for i in range(op_length):
                    if random.randint(0, 99) < np_chance:
                        operations.append(('np'))
                    else:
                        operations.append(('cp'))
                # sum up all the np for checking
                # print(f"Number of np operations: {operations.count('np')}")

                start_time = time.perf_counter()

                for op in operations:
                    if op == 'np' and current_device == 'gpu':
                        current_array = cp.asnumpy(current_array)
                        current_device = 'cpu'
                    elif op == 'cp' and current_device == 'cpu':
                        current_array = cp.array(current_array)
                        current_device = 'gpu'
                    if op == 'np':
                        result = np_compute(current_array, current_array, a_val_np, b_val_np, c_val_np)
                    else:
                        result = cp_compute(current_array, current_array, a_val_cp, b_val_cp, c_val_cp)

                cp.cuda.Stream.null.synchronize()
                end_time = time.perf_counter()
                with open("runtime.log", "a") as f:
                    f.write(f"{array_length},{op_length},{np_chance},{end_time - start_time}\n")
            with open("runtime.log", "a") as f:
                f.write("\n")
    return null