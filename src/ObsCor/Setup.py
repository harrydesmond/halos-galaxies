#!/usr/bin/env python3
# coding: utf-8
# Script that is going to hold some constants and useful functions

import math

# Set constants
h = 1
nside = 16
lims = {'min_z' : 0.005, 'max_z' : 0.064,
        'min_msol' : 9.8, 'max_msol' : 30,
        'min_ra' : 100, 'max_ra' : 300,
        'min_mag' : -100, 'max_mag' : 100}

# Random catalog settings
rand_size_mult = 1

# Corrfunc settings
nbins = 25
pimax = 40
cosmology = 2
nthreads = 10



# Useful functions
def classify(M_sol, RA, Z, MAG, lims):
    """
    Function that cuts the galaxy catalog based on some properties.
    Can do cuts in: Msol, RA, Z, MAG
    Mass input is not in log but the limits in log10
    """
    # Unpack the limits
    min_z, max_z = lims['min_z'], lims['max_z']
    min_ra, max_ra = lims['min_ra'], lims['max_ra'] # assumed in degrees
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
    
    if not (min_z < Z < max_z):
        return False

    if math.isinf(MAG):
        return False
    
    if not (min_mag < MAG < max_mag):
        return False
    return True
