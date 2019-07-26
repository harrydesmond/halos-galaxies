#!/usr/bin/env python3
# coding: utf-8 
import healpy as hp
import numpy as np
import Corrfunc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pickle
import kmeans_radec
import Setup as p

# Read the galaxy catalog
galaxy_catalog = np.load("../../Data/sdss_cutoff.npy")
RA, DEC, dist = p.unpack_catalog(galaxy_catalog)
N = RA.size

# Read the supplied randoms catalog
random_catalog = np.load("../../Data/randCat_matchnsa.npy")
rand_RA, rand_DEC, rand_Dist = p.unpack_catalog(random_catalog)
rand_N = rand_RA.size

# Setup the bins
bins = np.logspace(np.log10(p.min_rp), np.log10(p.max_rp), p.nbins + 1)

# Now assign each galaxy to a kmeans cluster that were precomputed on the random data set
X = np.vstack([RA, DEC]).T
with open("../../Data/km_clusters.p", 'rb') as handle:
    km = pickle.load(handle)

rand_gal_labels = km.labels
gal_labels = km.find_nearest(X)

print("Everything loaded")

# Let's do the heavy work..
def generate_wp(kcent, nthreads):
    # For this k-means center omit all galaxies

    # Do this for SDSS cat
    IDS = np.where(gal_labels != kcent)
    cRA = RA[IDS]
    cDEC = DEC[IDS]
    cDist = dist[IDS]
    cN = dist.size

    # Do the same as above but for the simulated catalog
    IDS = np.where(rand_gal_labels != kcent)
    crand_RA = rand_RA[IDS]
    crand_DEC = rand_DEC[IDS]
    crand_Dist = rand_Dist[IDS]
    crandN = crand_Dist.size

    # Make some plots to check if its doing the right thing
    hp.mollview(np.zeros(12), rot=180)
    hp.projscatter(np.pi/2-np.deg2rad(crand_DEC), np.deg2rad(crand_RA), s=0.01)
    plt.savefig("../../Plots/Clusters/{}Rand.png".format(kcent), dpi=180)
    plt.close()


    hp.mollview(np.zeros(12), rot=180)
    hp.projscatter(np.pi/2-np.deg2rad(cDEC), np.deg2rad(cRA), s=0.01)
    plt.savefig("../../Plots/Clusters/{}SDSS.png".format(kcent), dpi=180)
    plt.close()

    # Auto pair counts in DD
    autocorr = 1
    DD_counts = Corrfunc.mocks.DDrppi_mocks(autocorr, p.cosmology, nthreads,
                        p.pimax, bins, cRA, cDEC, cDist, is_comoving_dist=True)
    
    # Cross pair counts in DR
    autocorr = 0
    DR_counts = Corrfunc.mocks.DDrppi_mocks(autocorr, p.cosmology, nthreads,
                        p.pimax, bins, RA, DEC, dist, RA2=rand_RA,
                        DEC2=rand_DEC, CZ2=rand_Dist, is_comoving_dist=True)
    
    # Auto pairs counts in RR
    autocorr=1
    RR_counts = Corrfunc.mocks.DDrppi_mocks(autocorr, p.cosmology, nthreads,
                        p.pimax, bins, rand_RA, rand_DEC, rand_Dist,
                        is_comoving_dist=True)
    
    # All the pair counts are done, get the angular correlation function
    wp = Corrfunc.utils.convert_rp_pi_counts_to_wp(N, N, rand_N, rand_N,
            DD_counts, DR_counts, DR_counts, RR_counts, p.nbins, p.pimax)
    return wp

print("Starting pixel computation..")
wp_out = list()
start_time = time.time()
for kcent in range(p.ncent):
    wp = generate_wp(kcent, p.nthreads)
    print("Calculated w_p after covering cent {}".format(kcent))
    wp_out.append(wp)

wp_out = np.array(wp_out)
print("Finished in {}".format(time.time()-start_time))

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

# Calculate the STD on elements
std = np.sqrt(np.diagonal(cov_matrix))

# Save the output
output = dict()
for name, dat in zip(["cbins", "mean_wp", "covmap_wp"], [cbins, mean_wp, cov_matrix]):
    output[name] = dat

with open("../../Data/Obs_CF.p", 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open("../../Data/Obs_CFsubsamples.p", 'wb') as handle:
    pickle.dump(wp_out, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Let's make a plot!
fig = plt.figure()
ax = plt.axes()
ax.set_xscale("log")
ax.set_yscale("log")
ax.errorbar(cbins, mean_wp, yerr=std, label="Observational CF")
ax.plot(p.xr, p.yr, label='Data from Reddick', marker='o')
ax.set_xlabel(r'$r_p$')
ax.set_ylabel(r'$w_p$')
ax.set_ylim(bottom=10**(-1), top=10**(4))
ax.legend()
plt.tight_layout()   
plt.savefig("../../Plots/Corrfunc/4_CFcompar.png", dpi=180)
plt.close()
