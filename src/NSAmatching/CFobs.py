#!/usr/bin/env python3
# coding: utf-8 
import healpy as hp
import numpy as np
import Corrfunc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from time import time
import sys
import pickle
import kmeans_radec
import Setup as p

# Parse the input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--threads", dest="threads", help="Number of threads",
                    type=str, default=8)
parser.add_argument("--logmslim", dest="logMSlim", help="Lower limit on stellar mass",
                    type=str, default=None)
args = parser.parse_args()
ncores= int(args.threads)
logSMlim = float(args.logMSlim)

# Read the galaxy catalog
galaxy_catalog = np.load("../../Data/NSAmatching/CFcatNSA_{}_.npy".format(logSMlim))
RA, DEC, CZ = p.unpack_catalog(galaxy_catalog)
weights = galaxy_catalog['weights']
N = RA.size

# Read the supplied randoms catalog
random_catalog = np.load("../../Data/NSAmatching/CFrandcatNSA_{}_.npy".format(logSMlim))
rand_RA, rand_DEC, rand_CZ = p.unpack_catalog(random_catalog)
rand_weights = galaxy_catalog['weights']
rand_N = rand_RA.size

# Setup the bins
bins = np.logspace(np.log10(p.min_rp), np.log10(p.max_rp), p.nbins + 1)


rand_gal_labels = random_catalog['label']
gal_labels = galaxy_catalog['label']

print("Everything loaded")
sys.stdout.flush()

# Let's do the heavy work..
def generate_wp(kcent, nthreads):
    # For this k-means center omit all galaxies

    # Do this for SDSS cat
    IDS = np.where(gal_labels != kcent)
    cRA = RA[IDS]
    cDEC = DEC[IDS]
    cCZ= CZ[IDS]
    cweights = weights[IDS]
    cN = cCZ.size

    # Do the same as above but for the simulated catalog
    IDS = np.where(rand_gal_labels != kcent)
    crand_RA = rand_RA[IDS]
    crand_DEC = rand_DEC[IDS]
    crand_CZ= rand_CZ[IDS]
    crand_weights = rand_weights[IDS]
    crandN = crand_CZ.size

    # Auto pair counts in DD i.e. survey catalog
    autocorr = 1
    DD_counts = Corrfunc.mocks.DDrppi_mocks(autocorr, p.cosmology, nthreads,
                        p.pimax, bins, cRA, cDEC, cCZ, cweights,
                        weight_type='pair_product')
    
    # Cross pair counts in DR
    autocorr = 0
    DR_counts = Corrfunc.mocks.DDrppi_mocks(autocorr, p.cosmology, nthreads,
                        p.pimax, bins, cRA, cDEC, cCZ, cweights, RA2=crand_RA,
                        DEC2=crand_DEC, CZ2=crand_CZ, weights2=crand_weights,
                        weight_type='pair_product')
    
    # Auto pairs counts in RR i.e. random catalog
    autocorr=1
    RR_counts = Corrfunc.mocks.DDrppi_mocks(autocorr, p.cosmology, nthreads,
                        p.pimax, bins, crand_RA, crand_DEC, crand_CZ,
                        crand_weights, weight_type='pair_product')
    
    # All the pair counts are done, get the angular correlation function
    wp = Corrfunc.utils.convert_rp_pi_counts_to_wp(cN, cN, crandN, crandN,
            DD_counts, DR_counts, DR_counts, RR_counts, p.nbins, p.pimax)
    return wp

print("Starting pixel computation..")
wp_out = list()
extime = list()

for kcent in range(p.ncent):
    start = time()
    wp = generate_wp(kcent, ncores)
    wp_out.append(wp)

    t = time()-start
    extime.append(t)
    remtime = sum(extime)/len(extime)*(p.ncent-kcent-1)/60**2
    print("Done with step {}/{} in time {:.1f}. Estimated remaining time is {:.2f} hours".format(1+kcent, p.ncent, t, remtime))
    sys.stdout.flush()

wp_out = np.array(wp_out)

Nsub = p.ncent
# The shape is Nsubsample x nbins
mean_wp = np.mean(wp_out, axis=0)
ndim = mean_wp.size
cov_matrix = np.zeros(shape=(ndim, ndim))
for i in range(ndim):
    for j in range(ndim):
        for k in range(Nsub):
            cov_matrix[i, j] += (wp_out[k, i]-mean_wp[i])*(wp_out[k, j]-mean_wp[j])
cov_matrix = cov_matrix*(Nsub-1)/Nsub
cbins = p.bin_centers(bins)

# Save the output
output = dict()
for name, dat in zip(["cbins", "mean_wp", "covmap_wp"], [cbins, mean_wp, cov_matrix]):
    output[name] = dat


# Save the pickles
p.dump_pickle(output, "../../Data/NSAmatching/Obs_CF_SMcut_{}_.p".format(logSMlim))

print("Finished everything!")
sys.stdout.flush()
