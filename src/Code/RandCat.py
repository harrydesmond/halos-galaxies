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
import kmeans_radec
import pickle

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
galaxy_catalog = np.load("../../Data/sdss_cutoffV2.npy")
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

start_time = time.time()

# Total number of tasks to run over all procs
Nmock = Dist.size*p.rand_size_mult                  
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
    comm.Send(buff, dest=0)

if rank == 0:                           # At end, a single thread does things like concatenate the files
    string = 'cat `find ./ -name "out_*" | sort -V` > out.dat'
    os.system(string)
    string = 'rm out_*.dat'
    os.system(string)

    end_time = time.time()
    print("Generating the random catalog took me {} seconds".format(end_time-start_time))
    
    out = np.loadtxt("out.dat")
    os.system("rm out.dat")
    
    DEC = np.rad2deg(out[:, 0])
    RA = np.rad2deg(out[:, 1])
    dist = out[:, 2]
    
    
    random_catalog = np.zeros(Nmock, dtype={'names':('ra', 'dec', 'dist'),
                                'formats':('float64', 'float64', 'float64')})
    random_catalog['ra'] = np.ravel(RA)
    random_catalog['dec'] = np.ravel(DEC)
    random_catalog['dist'] = np.ravel(dist)

    # Now calculate the k-means centers 
    X = np.vstack([RA, DEC]).T
    km = kmeans_radec.kmeans_sample(X, p.ncent, maxiter=250, tol=1.0e-5)
    # Save the km object
    p.dump(km, "../../Data/km_clusters.p")
    
    hp.mollview(np.zeros(12), rot=180)
    for lab in range(p.ncent):
        IDS = np.where(km.labels == lab)
        hp.projscatter(np.pi/2-np.deg2rad(DEC[IDS]), np.deg2rad(RA[IDS]), s=1)
    plt.savefig("../../Plots/Corrfunc/Clustering.png", dpi=240)
    plt.close()
    
    np.save('../../Data/randCat_matchnsa.npy', random_catalog)
    
    print("Done with generating the catalog of size {}!".format(RA.size))


    
   # Make plots to check everything is going acccording to plan :-O
    plt.figure()
    plt.hist(dist, bins='auto')
    plt.savefig("../../Plots/Corrfunc/3_Hist_randDist.png")
    plt.close()
    
    print("And don't forget the HealPy plot!")  
    # And a HealPy map to plot galaxies pos..
    hp.mollview(np.zeros(12), rot=[180, 0, 0])
    hp.projscatter(np.pi/2-np.deg2rad(DEC), np.deg2rad(RA), s=0.001, c='red')
    plt.savefig("../../Plots/Corrfunc/3_Mollview_rand_cat.png")
    plt.close() 
