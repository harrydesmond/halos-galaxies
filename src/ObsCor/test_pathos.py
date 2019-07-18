#!/usr/bin/env python3
# coding: utf-8
import pathos
import numpy as np
import math
import scipy.stats as stats
import time
import multiprocessing as mp

start_time = time.time()

npools = 4
nmax = int(1e3)

print("Running with {} pools for {} times".format(npools, nmax))

pool = pathos.pools.ProcessPool(npools)

def get_number(i):
    while True:
        x, y, z = stats.uniform.rvs(loc=0, scale=1, size=3)*10

        dist = math.sqrt(x**2+y**2+z**2)

        if dist < 1:
            return dist





loop_out = pool.amap(get_number, range(nmax))
end_time = time.time()
pool.close()

print(end_time-start_time)

#print("Output size is {}".format(len(loop_out)))