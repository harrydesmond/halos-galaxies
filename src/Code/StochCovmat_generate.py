#!/usr/bin/env python3
# coding: utf-8 
import numpy as np
from time import time
import Setup as p
import Likelihood

# Initiate my likelihood model
model = Likelihood.Posterior()
ncore = 6

# How dense should the grid be
Nalphas = 100
Nscatters = 15
alphas = np.linspace(p.min_alpha, p.max_alpha, Nalphas)
scatters = np.linspace(p.min_scatter, p.max_scatter, Nscatters)

# Say x-dimension corresponds to alpha, y-dimension corresponds to scatter
XX, YY = np.meshgrid(alphas, scatters)
ndim1, ndim2 = XX.shape

# Calculate the stochastic covariance matrix at these values
Niter = 75
ntot = XX.size

means = np.zeros(shape=(ndim1, ndim2, p.nbins))
covmats = np.zeros(shape=(ndim1, ndim2, p.nbins, p.nbins))

k = 1

for i in range(ndim1):
    for j in range(ndim2):
        start = time()
        alpha, scatter = XX[i, j], YY[i, j]
        catalogs = model.abundance_match(alpha, scatter, Niter)
        cov_matrix, mean = model.stoch_covmat_mean(catalogs, nthreads=ncore)

        means[i, j, :] = mean
        covmats[i, j, :, :] = cov_matrix
        

        print("Finished step {}/{} in {} seconds".format(k, ntot, time()-start))
        k += 1

res = {'alpha' : XX, 'scatter' : YY, 'covmat' : covmats, 'wp' : means} 

p.dump_pickle(res, "/mnt/zfsusers/rstiskalek/Data/Train_stoch_covmats.p")


print("Finished")
