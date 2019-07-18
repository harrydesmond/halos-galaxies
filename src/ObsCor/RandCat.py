#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import scipy.stats
import astropy.constants as consts
import pathos.multiprocessing
import math
import Corrfunc
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from astropy.io import fits

import Setup as p
#########################################
#### Start generating random catalog ####
#########################################

# Set distance limits based on the cut catalog
min_dist = np.min(Dist)
max_dist = np.max(Dist)
boxsize = max_dist

# Pools cause threading is fun!!

pool = pathos.multiprocessing.ProcessingPool(N_pools)

def get_galaxy(i):
    while True:
        x, y, z = scipy.stats.uniform.rvs(loc=-1, scale=2, size=3)*boxsize
        pix = hp.vec2pix(p.nside, x,y,z)

        if pix in pixs_list:
            sim_dist = np.sqrt(x**2+y**2+z**2)
            if min_dist < sim_dist < max_dist:
                theta, phi = hp.vec2ang(np.array([x,y,z]))
                return [np.pi/2-theta, phi, sim_dist]

# Set how large the rand catalog should be
Nmax = Dist.size*rand_size_mult

print("Generating a random catalog of size {:d}".format(Nmax))

loop_out = pool.map(get_galaxy, range(Nmax))
rand_DEC = np.array([float(item[0]) for item in loop_out])
rand_RA = np.array([float(item[1]) for item in loop_out])
rand_Dist = np.array([float(item[2]) for item in loop_out])

# Close the pools
pool.close()
pool.join()
pool.clear()
    

random_catalog = np.zeros(Nmax, dtype={'names':('ra', 'dec', 'dist'),
                            'formats':('float64', 'float64', 'float64')})
random_catalog['ra'] = np.rad2deg(np.ravel(rand_RA))
random_catalog['dec'] = np.rad2deg(np.ravel(rand_DEC))
random_catalog['dist'] = np.ravel(rand_Dist)


np.save('../../Data/randCat_matchnsa.npy', random_catalog)

print("Done with generating the catalog!")
