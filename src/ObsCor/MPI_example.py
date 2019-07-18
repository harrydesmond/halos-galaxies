#!/usr/bin/env python3
# coding: utf-8 
import numpy as np  
import math
import scipy.stats
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import os
import sys

import Setup as p

# Setup MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
MPI_size = comm.Get_size()

# Load galaxy catalog
galaxy_catalog = np.load("../../Data/sdss_cutoff.npy")
pixs_list = np.load("../../Data/gpixs_list.npy")

# Set distance limits based on the cut catalog
Dist = galaxy_catalog["dist"]
min_dist = np.min(Dist)
max_dist = np.max(Dist)

def get_galaxy(boxsize):
    while True:
        x, y, z = scipy.stats.uniform.rvs(loc=-1, scale=2, size=3)*boxsize
        pix = hp.vec2pix(p.nside, x,y,z)

        if pix in pixs_list:
            sim_dist = np.sqrt(x**2+y**2+z**2)
            if min_dist < sim_dist < max_dist:
                theta, phi = hp.vec2ang(np.array([x,y,z]))
                return [np.pi/2-theta, phi, sim_dist]

Nmock = int(1e2)                                     # Total number of tasks to run over all procs

N_per_proc = int(np.floor(Nmock/MPI_size))          # Divide tasks between procs

if rank != MPI_size-1:
    N_proc = N_per_proc
else:
    N_proc = Nmock - (MPI_size-1)*N_per_proc    # Last proc does slightly more if MPI_size doesn't divide Nmock

output = list()
for i in range(N_proc):
    output.append(get_galaxy(boxsize=max_dist))


out_file = 'out_'+str(rank)+'.dat'                  # Label output file by proc rank
np.savetxt(out_file, output)

buff = np.zeros(1)                      # This causes threads to wait until they've all finished
if rank==0:
    for i in range(1, MPI_size):
        comm.Recv(buff, source=i)
else:
    comm.Send(buff)

if rank == 0:                           # At end, a single thread does things like concatenate the files
    string = 'cat `find ./ -name "out_*" | sort -V` > out.dat'
    os.system(string)
    string = 'rm out_*.dat'
    os.system(string)

#random_catalog = np.zeros(Nmax, dtype={'names':('ra', 'dec', 'dist'),
#                            'formats':('float64', 'float64', 'float64')})
#random_catalog['ra'] = np.rad2deg(np.ravel(rand_RA))
#random_catalog['dec'] = np.rad2deg(np.ravel(rand_DEC))
#random_catalog['dist'] = np.ravel(rand_Dist)
#
#
#np.save('../../Data/randCat_matchnsa.npy', random_catalog)
#
#print("Done with generating the catalog!")

