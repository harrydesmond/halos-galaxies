#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
import Corrfunc
import pickle
import Setup as p
from time import time
# Later I might want to generate the catalogs directly here # for now let's just load it and figure out the jackknifing 

# Load the AM matched catalof and cut some parts off
catalog = np.load("../../Catalog_Baryon/Catalog_BMF_Iter0_Alpha0,0.npy")

# These cuts will have to be later edited as well and prob apply cut in SM

sm = catalog['cat16']
IDS = np.where(sm>9.8)
x = catalog['x'][IDS]
y = catalog['y'][IDS]
z = catalog['z'][IDS]

rp_bins = np.logspace(np.log10(p.min_rp), np.log10(p.max_rp), p.nbins+1)

# Split the box into subvolumes

Nsub = 256
edges = np.arange(0, 400+25, 25)
box_labels = list()
# Figure out into which boxes galaxies belong
lx = np.digitize(x, edges)-1
ly = np.digitize(y, edges)-1

side = 16
gbins = ly*side+lx
print("This many gals {}".format(x.size))
# Do some computation
wp_out = list()
for i in range(side**2):
    IDS = np.where(gbins != i)

    XX = x[IDS]
    YY = y[IDS]
    ZZ = z[IDS]
    print("Entering w_p calc")
    start = time()
    wp = Corrfunc.theory.wp(boxsize=400, pimax=p.pimax, nthreads=p.nthreads, binfile=rp_bins, X=XX, Y=YY, Z=ZZ)
    wp_out.append(wp['wp'])
    print("Done with {}/{} in {} seconds.".format(i, (edges.size-1)**2, time()-start))

wp_out = np.array(wp_out)

with open("../../Data/test_simCF.p", 'wb') as handle:
    pickle.dump(wp_out, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("Done.. the final shape is: {}".format(wp_out.shape))
mean_wp = np.mean(wp_out, axis=0)
ndim = mean_wp.size
cov_matrix = np.zeros((ndim, ndim))
for i in range(ndim):
    for j in range(ndim):
        for k in range(side**2):
            cov_matrix[i, j] += (wp_out[k, i]-mean_wp[i])*(wp_out[k, j]-mean_wp[j])

cov_matrix = cov_matrix*(Nsub-1)/Nsub
cbins = p.bin_centers(rp_bins)

std = np.sqrt(np.diagonal(cov_matrix))

# This one should later be averaged in bins
fig = plt.figure()
ax = plt.axes()
ax.set_xscale("log")
ax.set_yscale("log")
ax.errorbar(cbins, mean_wp, yerr=std, label="Simulated CF")
ax.plot(p.xr, p.yr, label='Data from Reddick', marker='o')
ax.set_xlabel(r'$r_p$')
ax.set_ylabel(r'$w_p$')
ax.set_ylim(bottom=10**(-1), top=10**(4))
ax.legend()
plt.tight_layout()   
plt.savefig("../../Plots/Corrfunc/4_CFcomparSIM.png", dpi=180)
plt.close()

