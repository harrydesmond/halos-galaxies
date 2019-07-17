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
from astropy.io import fits

# Set constants
h = 0.688
nside = 14
lims = {'min_z' = 0, 'max_z' = 0.064
        'min_msol' = 9.8, 'max_msol' = 30,
        'min_ra' = 100, 'max_ra' = 300,
        'min_mag' = -100, 'max_mag' = 100}

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
        
    if not (min_msol < np.log10(M_sol) < max_msol):
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

survey = fits.open('../BAM/Old/nsa_v1.fits')[1].data
mass_survey = survey['SERSIC_MASS']/h**2
RA_survey = survey['RA'] # in degrees
DEC_survey = survey['DEC'] # in degrees
Z_survey = survey['ZDIST']
MAG_r = survey['SERSIC_ABSMAG'][:, 4]

# Make original plots of the survey data
hp.mollview(np.zeros(12), rot=[180, 0, 0])
hp.projscatter(np.pi/2-np.deg2rad(DEC_survey), np.deg2rad(RA_survey), s=0.001, c='red')
plt.savefig("../Plots/CorrFunc/Mollview_survey.png", dpi=180)
plt.close()

plt.figure()
plt.hist(Z_survey, bins='auto')
plt.savefig("../Plots/Corrfunc/Hist_Z_survey.pdf", dpi=180)
plt.close()

plt.figure()
plt.hist(RA_survey, bins='auto')
plt.savefig("../Plots/Corrfunc/Hist_RA_survey.pdf", dpi=180)
plt.close()

# Perform cuts in Z, msol, and RA
IDS = list()

for m, r, d, rs, mag, i in zip(mass_survey, RA_survey, DEC_survey,
                            Z_survey, MAG_r, np.arange(mass_survey.size)):
    if classify(m, r, rs, mag, lims) == True:
        IDS.append(i)
        
IDS = np.array(IDS)

RA = RA_survey[IDS]
DEC = DEC_survey[IDS]
Dist = Z_survey[IDS]*consts.c.to_value('km/s')/(h*100)
log_mass = np.log10(mass_survey[IDS])

# Count galaxies in pixels
pix_count = np.zeros(hp.nside2npix(nside))

for r, d in zip(np.deg2rad(RA), np.deg2rad(DEC)):
    pix = hp.ang2pix(nside, np.pi/2-d, r)
    pix_count[pix] += 1

#Get a list of pixels in which there is a sufficient number of galaxies.
# Furthermore check if its neighbours also have galaxies to eliminate outliers
active_pixels = np.zeros(hp.nside2npix(nside))
pixs_list = list()
for i in range(pix_count.size):
    # This cut was set by experimenting with the data, might have to be replaced
    if pix_count[i] > 300:
        pixs_list.append(i)
        active_pixels[i] = 1
# Counts the neighbours to make sure there are no holes. If more than 3 neighbours of a pixel also
# have galaxies then add to the list :)
for i in range(active_pixels.size):
    ngbh = hp.get_all_neighbours(nside, i)
    count = 0
    for ng in ngbh:
        if (ng in pixs_list) == False:
            count += 1
    if count <= 3:
        active_pixels[i] = 1
        pixs_list.append(i)

# Make a mollview map of pixels over which random catalog will be distributed

hp.mollview(active_pixels, rot=[180, 0, 0], min=-1, max=1)
hp.projscatter(np.pi/2-np.deg2rad(DEC), np.deg2rad(RA), s=0.001, c='red', alpha=0.3)
plt.savefig("../Plots/Corrfunc/Mollview_PixSelect")
plt.close()

# Plot the distribution of cuts

plt.figure()
plt.hist(Dist, bins='auto')
plt.savefig("../Plots/Corrfunc/Hist_Dist_cut.pdf", dpi=180)
plt.close()

plt.figure()
plt.hist(log_mass, bins='auto')
plt.savefig("../Plots/Corrfunc/Hist_Mass_cut.pdf", dpi=180)
plt.close()

# Save the cut galaxy catalog
N = RA.size
cut_survey = np.zeros(N, dtype={'names':('ra', 'dec', 'dist'),
                          'formats':('float64', 'float64', 'float64')})
cut_survey['ra'] = np.ravel(RA)
cut_survey['dec'] = np.ravel(DEC)
cut_survey['dist'] = np.ravel(Dist)
np.save('../Data/sdss_cutoff.npy', cut_survey)


## Start generating random catalog
#N_pools = 8
#min_dist = np.min(Dist)
#max_dist = np.max(Dist)
#boxsize=max_dist
#
#
#
#pool = pathos.multiprocessing.ProcessingPool(N_pools)
#
#def get_galaxy(i):
#    while True:
#        x, y, z = scipy.stats.uniform.rvs(loc=-1, scale=2, size=3)*boxsize
#        pix = hp.vec2pix(nside, x,y,z)
#
#        if pix in pixs_list:
#            sim_dist = np.sqrt(x**2+y**2+z**2)
#            if min_dist < sim_dist < max_dist:
#                theta, phi = hp.vec2ang(np.array([x,y,z]))
#                return [np.pi/2-theta, phi, sim_dist]
##                 return [np.pi/2-theta, phi, sim_dist, x, y, z]
#
#Nmax = Dist.size*100
## Nmax = 50000
#
#
#
#loop_out = pool.map(get_galaxy, range(Nmax))
#rand_DEC = np.array([float(item[0]) for item in loop_out])
#rand_RA = np.array([float(item[1]) for item in loop_out])
#rand_Dist = np.array([float(item[2]) for item in loop_out])
#
#
#pool.close()
#pool.join()
#pool.clear()
#    
#
#catalog = np.zeros(Nmax, dtype={'names':('ra', 'dec', 'dist'),
#                          'formats':('float64', 'float64', 'float64')})
#catalog['ra'] = np.rad2deg(np.ravel(rand_RA))
#catalog['dec'] = np.rad2deg(np.ravel(rand_DEC))
#catalog['dist'] = np.ravel(rand_Dist)
#
#
#np.save('../Data/randCat_matchnsa.npy', catalog)
#
#
## In[25]:
#
#
## Make a histogram just to check everything is going according to plan..
#plt.figure(dpi=120)
#plt.hist(Dist, bins='auto', density=1, label='gal catalog')
#plt.hist(rand_Dist, bins='auto', density=1, alpha=0.5, label='random')
#plt.legend()
#plt.tight_layout()
#plt.savefig("../Plots/Fig5_DistHist.pdf")
#plt.show()
#
## And a HealPy map to plot galaxies pos..
#hp.mollview(np.zeros(hp.nside2npix(nside)), rot=[180, 0, 0])
#hp.projscatter(np.pi/2-rand_DEC, rand_RA, s=0.0001, c='blue', alpha=0.2)
#hp.projscatter(np.pi/2-np.deg2rad(DEC), np.deg2rad(RA), s=0.05, c='r')
#plt.savefig("../Plots/Fig6_GalandRand")
#plt.show()
#
#
## In[26]:
#
#
#rand_RA.size
#
#
## In[ ]:
#
#
#
#
