#!/usr/bin/env python3
# coding: utf-8
# Script that is going to hold some constants and useful functions
import math
import numpy as np
import pickle
import joblib

# Set constants
h = 1
nside = 36
lims = {'min_m' : 9.4, 'max_m' : 30,
        'min_ra' : 100, 'max_ra' : 300,
        'min_dec' : -4.8, 'max_dec' : 75}

# MCMC stuff
Nens = 100

# Random catalog settings
rand_size_mult = 150
min_alpha = 0.0
max_alpha = 1.25
min_scatter = 0.0
max_scatter = 0.3

boxsize = 400
subside = 25

# Corrfunc settings
nbins = 30
min_rp = 0.1
max_rp = 30
ncent = 100

pimax = 40
cosmology = 2
nthreads = 8

# Useful functions
def classify(M_sol, RA, DEC, lims):
    """
    Function that cuts the galaxy catalog based on some properties.
    Can do cuts in: M, RA, Z, MAG
    Mass input is not in log but the limits in log10
    """
    # Unpack the limits
    min_ra, max_ra = lims['min_ra'], lims['max_ra'] # assumed in degrees
    min_dec, max_dec = lims['min_dec'], lims['max_dec'] # assumed in degrees
    min_m, max_m = lims['min_m'], lims['max_m'] # assumed in log10
    
    flag = False
    if math.isinf(M_sol) == True:
        return False
    if not M_sol > 0:
        return False
        
    if not (min_m < math.log10(M_sol) < max_m):
        return False
   
    if not (min_ra < RA < max_ra):
        return False
    
    if not (min_dec < DEC < max_dec):
        return False
    return True


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
    with open(filename, 'rb') as handle:
        obj = joblib.load(handle)
    return obj
