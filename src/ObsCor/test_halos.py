#!/usr/bin/env python3
# coding: utf-8 
import numpy as np
# import matplotlib.pyplot as plt
import Setup as p
import Corrfunc
import Likelihood
import pickle

model = Likelihood.MasterEquation()

catalog = model.abundance_match(0.5, 0.16, 1)

import healpy as hp
import halotools.mock_observables

x = catalog[0]['x']
y = catalog[0]['y']
z = catalog[0]['z']

lbox = 400
npoints = x.size*10
xrand = np.random.uniform(0, lbox, npoints)
yrand = np.random.uniform(0, lbox, npoints)
zrand = np.random.uniform(0, lbox, npoints)

coords = np.vstack((x,y,z)).T
randoms = np.vstack((xrand,yrand,zrand)).T
rp_bins = np.logspace(-1, np.log10(30), 30+1)
pi_max = 40

wp, wp_cov = halotools.mock_observables.wp_jackknife(coords, randoms, rp_bins, pi_max, Nsub=5, period=lbox,
                                                    num_threads=8)

x = p.bin_centers(rp_bins)
res = {'x' : x, 'wp' : wp, 'wp_cov' : wp_cov}
with open("../../Data/HT_wptest.p", 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
