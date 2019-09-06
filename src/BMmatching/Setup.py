#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import pickle
import joblib

# Set constants
h = 0.688
nside = 25

min_alpha = -1.0
max_alpha = 0.5
min_scatter = 0.0
max_scatter = 0.30
Nalphas = 10
Nscattes = 10
grid_size = 5

boxsize = 400
subside = 25

# Corrfunc settings
nbins = 30
min_rp = 0.1
max_rp = 30
# all of thse assume units Mpc/h
bins = np.logspace(np.log10(min_rp), np.log10(max_rp), nbins+1)
ncent = 100

pimax = 40
cosmology = 2
nthreads = 8

def unpack_catalog(catalog):
    return catalog['ra'], catalog['dec'], catalog['dist']


def bin_centers(edges):
    cents = list()
    for i in range(len(edges)-1):
        cents.append((edges[i+1]+edges[i])/2)
    return np.array(cents)


def dump_pickle(obj, filename):
    """
    Saves a pickle. Inputs are obj and filename
    """
    with open(filename, 'wb') as handle:
        joblib.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
 

def load_pickle(filename):
    """
    Loads a pickle. Input filename.
    """
    try:
        with open(filename, 'rb') as handle:
            obj = joblib.load(handle)
        return obj
    except:
        print("File {} not found. This is ok if about to generate cov matrices".format(filename))
        return np.nan
