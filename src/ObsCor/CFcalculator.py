#!/usr/bin/env python3
# coding: utf-8 
import numpy as np
import healpy as hp
import Corrfunc
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Auto pair counts in DD
autocorr = 1
DD_counts = DDrppi_mocks(autocorr, p.cosmology, p.nthreads, p.pimax, bins, RA, DEC, dist, is_comoving_dist=True)

# Cross pair counts in DR
autocorr = 0
DR_counts = DDrppi_mocks(autocorr, p.cosmology, p.nthreads, p.pimax, bins, RA, DEC, dist, RA2=rand_RA,
                        DEC2=rand_DEC, CZ2=rand_Dist, is_comoving_dist=True)

# Auto pairs counts in RR
autocorr=1
RR_counts = DDrppi_mocks(autocorr, p.cosmology, p.nthreads, p.pimax, bins, rand_RA, rand_DEC, rand_Dist, is_comoving_dist=True)

# All the pair counts are done, get the angular correlation function
wp = convert_rp_pi_counts_to_wp(N, N, rand_N, rand_N, DD_counts, DR_counts, DR_counts, RR_counts, p.nbins, p.pimax)

# Let's make a plot!
wp_sim = np.load("../../Data/halocorr.npy")




plt.figure()
plt.loglog(x, wp, label='Observation')
plt.scatter(x, wp)
plt.loglog(xr, yr, label='Reddick')
plt.scatter(xr, yr)
plt.loglog(wp_sim['x'],wp_sim['wp'], label='Simulation')
plt.scatter(wp_sim['x'], wp_sim['wp'])
plt.ylim(ymin=1, ymax=2000)
plt.xlim(xmin=0.1, xmax=30)
plt.xlabel(r'$r_p$')
plt.ylabel(r'$w_p$')
plt.legend()
plt.tight_layout()
plt.savefig("../../Plots/Corrfunc/4_CFcompar.pdf", dpi=180)
plt.close()




