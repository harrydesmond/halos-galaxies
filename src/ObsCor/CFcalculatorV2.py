#!/usr/bin/env python3
# coding: utf-8 
import healpy as hp
import numpy as np
import Corrfunc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

#from mpi4py import MPI
#from mpi4py.MPI import ANY_SOURCE
#import os
#import sys

import Setup as p


## Setup MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#MPI_size = comm.Get_size()

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

pixs_list = np.load("../../Data/gpixs_list.npy")
# Find the pixels that are not on the boundary
inside_pixs = list()
for px in pixs_list:
    ngbhs = hp.get_all_neighbours(p.nside, px)
    flag = False
    for ngbh in ngbhs:
        if ngbh == -1:
            ngbh = hp.nside2npix(p.nside) -1 
        if not ngbh in pixs_list:
            flag = True

    if flag == False:
        inside_pixs.append(px)


# Pick which inside pixels will get eliminated (along with their neighbours)
chosen_pixels = np.random.choice(inside_pixs, size=p.Nsub)

# Convert RA, DEC into pix nums
gpixs = hp.ang2pix(p.nside, np.pi/2-np.deg2rad(DEC), np.deg2rad(RA))
rand_gpixs = hp.ang2pix(p.nside, np.pi/2-np.deg2rad(rand_DEC),
                        np.deg2rad(rand_RA))


# Let's do the heavy work..
def generate_wp(cpix, nthreads):
    masked_pixs = hp.get_all_neighbours(p.nside, cpix)
    masked_pixs = np.append(masked_pixs, cpix)

    # Pick which galaxies remain unmasked
    unmasked_gals = np.in1d(gpixs, masked_pixs)
    IDS = np.where(unmasked_gals==False)
    cRA = RA[IDS]
    cDEC = DEC[IDS]
    cDist = dist[IDS]
    cN = dist.size

    # Do the same as above but for the simulated catalog
    unmasked_gals = np.in1d(rand_gpixs, masked_pixs)
    IDS = np.where(unmasked_gals==False)
    crand_RA = rand_RA[IDS]
    crand_DEC = rand_DEC[IDS]
    crand_Dist = rand_Dist[IDS]
    crandN = crand_Dist.size

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
    print("Calculated w_p after covering pix {}".format(cpix))
    return wp

wp_out = list()
start_time = time.time()
for cpix in chosen_pixels:
    wp = generate_wp(cpix, p.nthreads)
    wp_out.append(wp)

wp_out = np.array(wp_out)
print("Finished in {}".format(time.time()-start_time))
print("Shape is {}".format(wp_out.shape))

# The shape is Nsubsample x nbins
mean_wp = np.mean(wp_out, axis=0)
print(mean_wp.shape, mean_wp, wp_out)

# Now do the statistics and return mean + covariance matrix



#
## Let's make a plot!
#wp_sim = np.load("../../Data/halocorr.npy")
#
#x = p.bin_centers(bins)
#
#
#plt.figure()
#plt.loglog(x, wp, label='Observation')
#plt.scatter(x, wp)
#plt.loglog(p.xr, p.yr, label='Reddick')
#plt.scatter(p.xr, p.yr)
#plt.loglog(wp_sim['x'],wp_sim['wp'], label='Simulation')
#plt.scatter(wp_sim['x'], wp_sim['wp'])
#plt.ylim(ymin=1, ymax=2000)
#plt.xlim(xmin=0.1, xmax=30)
#plt.xlabel(r'$r_p$')
#plt.ylabel(r'$w_p$')
#plt.legend()
#plt.tight_layout()
#plt.savefig("../../Plots/Corrfunc/4_CFcompar.pdf", dpi=180)
#plt.close()
#
#
#
#



