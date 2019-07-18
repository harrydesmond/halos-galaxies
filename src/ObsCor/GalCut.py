#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import astropy.constants as consts
import math
from astropy.io import fits

import Setup as p

# Import observational data
survey = fits.open('../../BAM/Old/nsa_v1.fits')[1].data
mass_survey = survey['SERSIC_MASS']/p.h**2
RA_survey = survey['RA'] # in degrees
DEC_survey = survey['DEC'] # in degrees
Z_survey = survey['ZDIST']
MAG_r = survey['SERSIC_ABSMAG'][:, 4] - 5*np.log10(p.h)

print("Loaded galaxy catalog data!")

# Make original plots of the survey data
hp.mollview(np.zeros(12), rot=[180, 0, 0])
hp.projscatter(np.pi/2-np.deg2rad(DEC_survey), np.deg2rad(RA_survey), s=0.001, c='red')
plt.savefig("../../Plots/Corrfunc/1_Mollview_survey.png")
plt.close()

plt.figure()
plt.hist(Z_survey, bins='auto')
plt.savefig("../../Plots/Corrfunc/1_Hist_Z_survey.png")
plt.close()

plt.figure()
plt.hist(RA_survey, bins='auto')
plt.savefig("../../Plots/Corrfunc/1_Hist_RA_survey.png")
plt.close()

# Perform cuts in Z, msol, and RA, MAG_r
IDS = list()
for m, r, rs, mag, i in zip(mass_survey, RA_survey, Z_survey, MAG_r, np.arange(mass_survey.size)):
    if p.classify(m, r, rs, mag, p.lims) == True:
        IDS.append(i)
IDS = np.array(IDS)

RA = RA_survey[IDS]
DEC = DEC_survey[IDS]
Dist = Z_survey[IDS]*consts.c.to_value('km/s')/(p.h*100)
log_mass = np.log10(mass_survey[IDS])
MAG_r = MAG_r[IDS]

# Count galaxies in pixels
pix_count = np.zeros(hp.nside2npix(p.nside))
for r, d in zip(np.deg2rad(RA), np.deg2rad(DEC)):
    pix = hp.ang2pix(p.nside, np.pi/2-d, r)
    pix_count[pix] += 1

# Get a list of pixels in which there is a sufficient number of galaxies.
# Furthermore check if its neighbours also have galaxies to eliminate outliers
# This settings might have to be tweaeked when choosing different limits
active_pixels = np.zeros(hp.nside2npix(p.nside))
pixs_list = list()
for i in range(pix_count.size):
    if pix_count[i] > 35:
        pixs_list.append(i)
        active_pixels[i] = 1

# If more than n neighbours are not active pixels then remove. Eliminates outliers
IDS = list()
for i in range(hp.nside2npix(p.nside)):
    neighbours = hp.get_all_neighbours(p.nside, i)
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
for i in range(hp.nside2npix(p.nside)):
    neighbours = hp.get_all_neighbours(p.nside, i)
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
    pix = hp.ang2pix(p.nside, np.pi/2-d, r)
    if pix in pixs_list:
        IDS.append(i)
IDS = np.array(IDS)

RA = RA[IDS]
DEC = DEC[IDS]
Dist = Dist[IDS]
log_mass = log_mass[IDS] 
MAG_r = MAG_r[IDS]

pix_area = hp.nside2pixarea(p.nside, degrees=True)
survey_area = pix_area*len(pixs_list)
N = Dist.size
print("After applying cuts, there are {:d} galaxies over an area of {:.2f} deg^2".format(N, survey_area))

# Make a mollview map of pixels over which random catalog will be distributed
hp.mollview(active_pixels, rot=[180, 0, 0], min=-1, max=1)
hp.projscatter(np.pi/2-np.deg2rad(DEC), np.deg2rad(RA), s=0.01, c='red', alpha=0.5)
plt.savefig("../../Plots/Corrfunc/2_Mollview_PixSelect.png", dpi=180)
plt.close()

# Plot the distribution of cuts
plt.figure()
plt.hist(Dist, bins='auto')
plt.savefig("../../Plots/Corrfunc/2_Hist_Dist_cut.png")
plt.close()

plt.figure()
plt.hist(log_mass, bins='auto')
plt.savefig("../../Plots/Corrfunc/2_Hist_Mass_cut.png")
plt.close()

plt.figure()
plt.hist(MAG_r, bins='auto')
plt.savefig("../../Plots/Corrfunc/2_Hist_Mag_cut.png")
plt.close()


# Let's calculate the apparent magnitude and plot it
m = MAG_r+5*np.log10(Dist*1e6)-5
apmagcut = 14

s = np.where(m < apmagcut)[0].size
print("There  are {:d} galaxies of apparent magnitude less than {:.2f}".format(s, apmagcut))

plt.figure()
plt.hist(m, bins='auto')
plt.savefig("../../Plots/Corrfunc/2_Hist_appMag_cut.png")
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

# Furthermore, save the active pixels
np.save("../../Data/gpixs_list.npy", pixs_list)

print("Done with cuts!")
