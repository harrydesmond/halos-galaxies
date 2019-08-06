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
lims = {'min_z' : 0.005, 'max_z' : 0.064,
        'min_msol' : 9.8, 'max_msol' : 30,
        'min_ra' : 100, 'max_ra' : 300,
        'min_dec' : -4.8, 'max_dec' : 75,
        'min_mag' : -100, 'max_mag' : 100}

# MCMC stuff
Nens = 100

# Random catalog settings
rand_size_mult = 75
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

# Reddick data (approximate...)
xr = [0.128622422089606, 0.20684122410707673, 0.3326403703585226,
      0.5350139833477285, 0.8605779048945061, 1.3960874510643801,
      2.2454469597021176, 3.641545236724509, 5.857244150005106,
      9.42373420397281, 15.176488599849602, 24.246492585165885]
yr = [512.3677407071057, 362.09840110670876, 253.06986529591367,
        171.06503647200338, 113.08892361578259, 70.7141679268513,
        47.79992948330511, 32.67094059733128, 21.83994077255245,
        13.505922697186861, 6.395183522903948, 2.7704555569535207]



# Useful functions
def classify(M_sol, RA, DEC, Z, MAG, lims):
    """
    Function that cuts the galaxy catalog based on some properties.
    Can do cuts in: Msol, RA, Z, MAG
    Mass input is not in log but the limits in log10
    """
    # Unpack the limits
    min_z, max_z = lims['min_z'], lims['max_z']
    min_ra, max_ra = lims['min_ra'], lims['max_ra'] # assumed in degrees
    min_dec, max_dec = lims['min_dec'], lims['max_dec'] # assumed in degrees
    min_msol, max_msol = lims['min_msol'], lims['max_msol'] # assumed in log10
    min_mag, max_mag = lims['min_mag'], lims['max_mag']
    
    flag = False
    if math.isinf(M_sol) == True:
        return False
    if not M_sol > 0:
        return False
        
    if not (min_msol < math.log10(M_sol) < max_msol):
        return False
   
    if not (min_ra < RA < max_ra):
        return False
    
    if not (min_dec < DEC < max_dec):
        return False

    if not (min_z < Z < max_z):
        return False

    if math.isinf(MAG):
        return False
    
    if not (min_mag < MAG < max_mag):
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
