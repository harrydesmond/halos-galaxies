#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from time import time
import sys
import Setup as p
import Likelihood
import argparse

# Parse the input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--threads", dest="threads", help="Number of threads",
                    type=str, default=8)
parser.add_argument("--perccat", dest="perccat", help="Sets how much of the catalog to exclude",
                    type=str, default=None)
args = parser.parse_args()
ncores= int(args.threads)
perccat = float(args.perccat)

cuts_def = p.load_pickle("../../Data/BMmatching/logMBcuts_def.p")
logBMlim = cuts_def[perccat]

# Initiate my likelihood model
model = Likelihood.Model(logBMlim, perccat, generator=True)
print("Initiated the model!")
sys.stdout.flush()

alphas = np.linspace(p.min_alpha, p.max_alpha, p.Nalphas)
scatters = np.linspace(p.min_scatter, p.max_scatter, p.Nscatters)

# Say x-dimension corresponds to alpha, y-dimension corresponds to scatter
XX, YY = np.meshgrid(alphas, scatters)
ndim1, ndim2 = XX.shape
# Calculate the stochastic covariance matrix at these values
Niter = 1
covmats = np.zeros(shape=(ndim1, ndim2, p.nbins, p.nbins))
Ntot = XX.size
k = 1
extime = list()

for i in range(ndim1):
    for j in range(ndim2):
        start = time()
        alpha, scatter = XX[i, j], YY[i, j]
        catalogs = model.abundance_match(alpha, scatter, Niter)
        cov_matrix = model.jackknife_sim(catalogs[0], ncores)[1]

        covmats[i, j, :, :] = cov_matrix

        t = time()-start
        extime.append(t)
        remtime = sum(extime)/len(extime)*(Ntot-k)/60**2
        print("Done with step {}/{} in time {:.1f}. Estimated remaining time is {:.2f} hours".format(k, Ntot, t, remtime))
        sys.stdout.flush()
        k += 1

res = {'alpha' : XX, 'scatter' : YY, 'covmat' : covmats}
p.dump_pickle(res, "../../Data/BMmatching/Train_jackknife_covmats_{}_.p".format(perccat))

print("Finished")
sys.stdout.flush()
