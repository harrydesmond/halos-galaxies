from pathos.pools import ProcessPool
import numpy as np
import scipy.stats
from time import time

def func(i):
    while True:
        x, y, z = scipy.stats.uniform.rvs(0, 10, size=3)
        if (x**2+y**2+z**2) < 1:
            return x

N = 150
start = time()
for i in range(N):
    func(i)
print("Singl thread completed in {}".format(time()-start))
start = time()
pool = ProcessPool(nodes=11)
results = pool.map(func, range(N))
print("Did threading in {}".format(time()-start))
