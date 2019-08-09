#!/usr/bin/env python3
# coding: utf-8 
import numpy as np
from time import time
import sys
import Setup as p
import Likelihood

# Initiate my likelihood model
model = Likelihood.Posterior()
print("Initiated the model")
sys.stdout.flush()
ncore = 6


# How dense should the grid be
Nalphas = 25
Nscatters = 25
alphas = np.linspace(p.min_alpha, p.max_alpha, Nalphas)
scatters = np.linspace(p.min_scatter, p.max_scatter, Nscatters)

# Say x-dimension corresponds to alpha, y-dimension corresponds to scatter
XX, YY = np.meshgrid(alphas, scatters)
ndim1, ndim2 = XX.shape

# Calculate the stochastic covariance matrix at these values
Niter = 40
Ntot = XX.size

means = np.zeros(shape=(ndim1, ndim2, p.nbins))
covmats = np.zeros(shape=(ndim1, ndim2, p.nbins, p.nbins))

k = 1
extime = list()
for i in range(ndim1):
    for j in range(ndim2):
        start = time()
        alpha, scatter = XX[i, j], YY[i, j]
        catalogs = model.abundance_match(alpha, scatter, Niter)
        cov_matrix, mean = model.stoch_covmat_mean(catalogs, nthreads=ncore)

        means[i, j, :] = mean
        covmats[i, j, :, :] = cov_matrix
        
        t = time()-start
        extime.append(t)
        remtime = sum(extime)/len(extime)*(Ntot-k)/60**2
        print("Done with step {}/{} in time {:.1f}. Estimated remaining time is {:.2f} hours".format(k, Ntot, t, remtime))
        sys.stdout.flush()

        k += 1

res = {'alpha' : XX, 'scatter' : YY, 'covmat' : covmats, 'wp' : means} 

p.dump_pickle(res, "../../Data/Train_stoch_covmats.p")


print("Finished")
sys.stdout.flush()
