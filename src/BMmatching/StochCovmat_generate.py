#!/usr/bin/env python3
# coding: utf-8 
import numpy as np
from time import time
import sys
from pathos.multiprocessing import ProcessPool
import argparse
import Setup as p
import os
import Likelihood

# Parse the input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--threads", dest="threads", help="Number of threads",
                    type=str, default=8)
parser.add_argument("--perccat", dest="perccat", help="Sets how much of the catalog to exclude",
                    type=str, default=None)
args = parser.parse_args()
ncore= int(args.threads)
perccat = float(args.perccat)

cuts_def = p.load_pickle("../../Data/BMmatching/logMBcuts_def.p")
logBMlim = cuts_def[perccat]

# Initiate my likelihood model
model = Likelihood.Model(logBMlim, perccat, generator=True)
print("Initiated the model")
sys.stdout.flush()


# How dense should the grid be
alphas = np.linspace(p.min_alpha, p.max_alpha, p.Nalphas)
scatters = np.linspace(p.min_scatter, p.max_scatter, p.Nscatters)

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
filename = "./Pickles/theta_{}_.p".format(logBMlim)

pool = ProcessPool(ncore)
for i in range(ndim1):
    for j in range(ndim2):
        start = time()

        alpha, scatter = XX[i, j], YY[i, j]
        p.dump_pickle([alpha, scatter], filename)
        def catfunc(i):
            alpha, scatter = p.load_pickle(filename)
            return model.abundance_match(alpha, scatter, 1)[0]
        
        catalogs = pool.map(catfunc, np.arange(Niter))
        os.system("rm " + filename)
        print("Generated catalogs") 
        #catalogs = model.abundance_match(alpha, scatter, Niter)
        cov_matrix, mean = model.stoch_covmat_mean(catalogs, nthreads=ncore)

        means[i, j, :] = mean
        covmats[i, j, :, :] = cov_matrix
        
        t = time()-start
        extime.append(t)
        remtime = sum(extime)/len(extime)*(Ntot-k)/60**2
        print("Done with step {}/{} in time {:.1f}. Estimated remaining time is {:.2f} hours".format(k, Ntot, t, remtime))
        sys.stdout.flush()

        k += 1

pool.close()
pool.join()
pool.terminate()
res = {'alpha' : XX, 'scatter' : YY, 'covmat' : covmats, 'wp' : means} 

p.dump_pickle(res, "../../Data/BMmatching/Train_stoch_covmats_{}_.p".format(perccat))


print("Finished")
sys.stdout.flush()
