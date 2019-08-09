#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import Setup as p
import Likelihood 
from time import time

model = Likelihood.Model()
print("Initiated the model.")
sys.stdout.flush()

Nalpha = 40
Nscatter = 40

alphas = np.linspace(p.min_alpha, p.max_alpha, Nalpha)
scatters = np.linspace(p.min_scatter, p.max_scatter, Nscatter)

alpha_grid, scatter_grid = np.meshgrid(alphas, scatters)
ll_grid = np.zeros(shape=(Nalpha, Nscatter))

k = 1
extime = list()
Ntot = ll_grid.size
for i in range(Nalpha):
    for j in range(Nscatter):
        start = time()
        alpha = alpha_grid[i, j]
        scatter = scatter_grid[i, j]
        theta = (alpha, scatter)
        ll_grid[i, j] = model.loglikelihood(theta)
        
        t = time()-start
        extime.append(t)
        remtime = sum(extime)/len(extime)*(Ntot-k)/60**2
        print("Done with step {}/{} in time {:.1f}. Estimated remaining time is {:.2f} hours".format(k, Ntot, t, remtime))
        sys.stdout.flush()
        k += 1

res = {'alpha' : alpha_grid, 'scatter' : scatter_grid, 'loglikelihood' : ll_grid}
p.dump_pickle(res, "../../Data/Grid_search.p")
