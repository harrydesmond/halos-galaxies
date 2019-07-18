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

print("Modules loaded. Starting!")

# Set constants
h = 1
nside = 16
lims = {'min_z' : 0.005, 'max_z' : 0.064,
        'min_msol' : 9.8, 'max_msol' : 30,
        'min_ra' : 100, 'max_ra' : 300,
        'min_mag' : -100, 'max_mag' : 100}
N_pools = 4
rand_size_mult = 1

nbins = 25
pimax = 40
cosmology = 2
nthreads = N_pools 

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

survey = fits.open('../../BAM/Old/nsa_v1.fits')[1].data
mass_survey = survey['SERSIC_MASS']/h**2
RA_survey = survey['RA'] # in degrees
DEC_survey = survey['DEC'] # in degrees
Z_survey = survey['ZDIST']
MAG_r = survey['SERSIC_ABSMAG'][:, 4] - 5*np.log10(h)

print("Loaded galaxy catalog data!")

# Make original plots of the survey data
hp.mollview(np.zeros(12), rot=[180, 0, 0])
hp.projscatter(np.pi/2-np.deg2rad(DEC_survey), np.deg2rad(RA_survey), s=0.001, c='red')
plt.savefig("../../Plots/Corrfunc/Mollview_survey.png")
plt.close()

plt.figure()
plt.hist(Z_survey, bins='auto')
plt.savefig("../../Plots/Corrfunc/Hist_Z_survey.png")
plt.close()

plt.figure()
plt.hist(RA_survey, bins='auto')
plt.savefig("../../Plots/Corrfunc/Hist_RA_survey.png")
plt.close()

# Perform cuts in Z, msol, and RA
IDS = list()

for m, r, rs, mag, i in zip(mass_survey, RA_survey, Z_survey, MAG_r, np.arange(mass_survey.size)):
    if classify(m, r, rs, mag, lims) == True:
        IDS.append(i)
        
IDS = np.array(IDS)

RA = RA_survey[IDS]
DEC = DEC_survey[IDS]
Dist = Z_survey[IDS]*consts.c.to_value('km/s')/(h*100)
log_mass = np.log10(mass_survey[IDS])
MAG_r = MAG_r[IDS]

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
    if pix_count[i] > 35:
        pixs_list.append(i)
        active_pixels[i] = 1
# If more than 5 neighbours are not active pixels then remove

IDS = list()
for i in range(hp.nside2npix(nside)):
    neighbours = hp.get_all_neighbours(nside, i)
    count = 0
    for ng in neighbours:
        if not ng in pixs_list:
            count += 1
    if count > 3:
        IDS.append(i)
active_pixels[IDS] = 0
for j in IDS:
    if j in pixs_list:
        pixs_list.remove(j)


# Check for holes. If there are any then add them to active pixels
IDS = list()
for i in range(hp.nside2npix(nside)):
    neighbours = hp.get_all_neighbours(nside, i)
    count = 0
    for ng in neighbours:
        if ng in pixs_list:
            count += 1
    if count > 4:
        IDS.append(i)

active_pixels[IDS] = 1
for j in IDS:
    if j not in pixs_list:
        pixs_list.append(j)



# Now, take galaxies only in the active pixels!
IDS = list()
for r, d, i in zip(np.deg2rad(RA), np.deg2rad(DEC), np.arange(RA.size)):
    pix = hp.ang2pix(nside, np.pi/2-d, r)
    if pix in pixs_list:
        IDS.append(i)

IDS = np.array(IDS)

RA = RA[IDS]
DEC = DEC[IDS]
Dist = Dist[IDS]
log_mass = log_mass[IDS] 
MAG_r = MAG_r[IDS]

pix_area = hp.nside2pixarea(nside, degrees=True)
survey_area = pix_area*len(pixs_list)
N = Dist.size
print("After applying cuts, there are {:d} galaxies over an area of {:.2f} deg^2".format(N,
    survey_area))

# Make a mollview map of pixels over which random catalog will be distributed
hp.mollview(active_pixels, rot=[180, 0, 0], min=-1, max=1)
hp.projscatter(np.pi/2-np.deg2rad(DEC), np.deg2rad(RA), s=0.01, c='red', alpha=0.5)
plt.savefig("../../Plots/Corrfunc/Mollview_PixSelect.png", dpi=160)
plt.close()

# Plot the distribution of cuts

plt.figure()
plt.hist(Dist, bins='auto')
plt.savefig("../../Plots/Corrfunc/Hist_Dist_cut.png")
plt.close()

plt.figure()
plt.hist(log_mass, bins='auto')
plt.savefig("../../Plots/Corrfunc/Hist_Mass_cut.png")
plt.close()

plt.figure()
plt.hist(MAG_r, bins='auto')
plt.savefig("../../Plots/Corrfunc/Hist_Mag_cut.png")
plt.close()


# Let's calculate the apparent magnitude and plot it
m = MAG_r+5*np.log10(Dist*1e6)-5
apmagcut = 14

s = np.where(m < apmagcut)[0].size
print("There  are {:d} galaxies of apparent magnitude less than {:.2f}".format(s, apmagcut))

plt.figure()
plt.hist(m, bins='auto')
plt.savefig("../../Plots/Corrfunc/Hist_appMag_cut.png")
plt.close()





# Save the cut galaxy catalog
N = RA.size
galaxy_catalog = np.zeros(N, dtype={'names':('ra', 'dec', 'dist', 'mag_r'),
                          'formats':('float64', 'float64', 'float64', 'float64')})
galaxy_catalog['ra'] = np.ravel(RA)
galaxy_catalog['dec'] = np.ravel(DEC)
galaxy_catalog['dist'] = np.ravel(Dist)
galaxy_catalog['mag_r'] = np.ravel(MAG_r)
np.save('../../Data/sdss_cutoff.npy', galaxy_catalog)

print("Done with cuts!")

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
        pix = hp.vec2pix(nside, x,y,z)

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




# Make plots to check everything is going acccording to plan :-O
plt.figure()
plt.hist(Dist, bins='auto', density=1, label='gal catalog')
plt.hist(rand_Dist, bins='auto', density=1, alpha=0.5, label='random')
plt.legend()
plt.tight_layout()
plt.savefig("../../Plots/Corrfunc/Hist_dist_randComp.png")
plt.close()


# And a HealPy map to plot galaxies pos..
hp.mollview(np.zeros(hp.nside2npix(nside)), rot=[180, 0, 0])
hp.projscatter(np.pi/2-rand_DEC, rand_RA, s=0.0001, c='blue', alpha=0.2)
hp.projscatter(np.pi/2-np.deg2rad(DEC), np.deg2rad(RA), s=0.05, c='r')
plt.savefig("../../Plots/Corrfunc/Mollview_rand_cat.png")
plt.close()


#############################################
#### Calculate the correlation function! ####
#############################################

print("Starting calculating CF")

def read_catalog(catalog):
    """Returns RA, DEC, DIST given some dictionary."""
    return catalog['ra'], catalog['dec'], catalog['dist']

# Get the galaxy catalog
RA, DEC, dist = read_catalog(galaxy_catalog)
N = RA.size

# Get the random catalog
rand_RA, rand_DEC, rand_Dist = read_catalog(random_catalog)
rand_N = rand_RA.size

# Setup the bins
bins = np.logspace(-1, np.log10(30), nbins+1)

# Auto-pair counts in DD
autocorr = 1
DD_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax,
                         bins, RA, DEC, dist, is_comoving_dist=True)

print("Finished auto-correlating gal. cat")

# Cross-pair counts in DR
autocorr = 0
DR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins,
                        RA, DEC, dist, RA2=rand_RA, DEC2=rand_DEC, CZ2=rand_Dist,
                        is_comoving_dist=True)

print("Finished cross-correlating")

# Auto-pair counts in RR
autocorr = 1
RR_counts = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins,
                         rand_RA, rand_DEC, rand_Dist, is_comoving_dist=True)

print("Finished auto-correlating rand. cat")

# All the pair counts are done, get the angular correlation function
wp = convert_rp_pi_counts_to_wp(N, N, rand_N, rand_N, DD_counts, DR_counts,
                                DR_counts, RR_counts, nbins, pimax)
print("Finished calculating the wp")

# Save calculated corr func
np.save("../../Data/obs_CF.npy", wp)

print("Saved the wp")

def bin_centers(edges):
    """ Calculates the centres of bins"""
    cents = list()
    for i in range(len(edges)-1):
        cents.append((edges[i+1]+edges[i])/2)
    return np.array(cents)

# Hardcoded Reddick results..

xr = [0.128622422089606, 0.20684122410707673, 0.3326403703585226,
      0.5350139833477285, 0.8605779048945061, 1.3960874510643801,
      2.2454469597021176, 3.641545236724509, 5.857244150005106,
      9.42373420397281, 15.176488599849602, 24.246492585165885]
yr = [512.3677407071057, 362.09840110670876, 253.06986529591367,
        171.06503647200338, 113.08892361578259, 70.7141679268513,
        47.79992948330511, 32.67094059733128, 21.83994077255245,
        13.505922697186861, 6.395183522903948, 2.7704555569535207]

# Load correlation function from a simulation

wp_sim = np.load("../../Data/halocorr.npy")

x = bin_centers(bins)

wp_sim = np.load("../../Data/halocorr.npy")


x = bin_centers(bins)


plt.figure()
plt.loglog(x, wp, label='Observation')
plt.scatter(x, wp)
plt.loglog(xr, yr, label='Reddick')
plt.scatter(xr, yr)
plt.loglog(wp_sim['x'],wp_sim['wp'], label='Simulation')
plt.scatter(wp_sim['x'], wp_sim['wp'])
plt.ylim(ymin=1, ymax=2000)
plt.xlim(xmin=0.1, xmax=30)
plt.xlabel(r'$r_p$')
plt.ylabel(r'$w_p$')
plt.legend()
plt.tight_layout()
plt.savefig("../../Plots/Corrfunc/CorrCompar.png", dpi=180)
plt.close()

print("Finished!!1")

