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
import pickle
from sklearn.cluster import DBSCAN

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

# Perform cuts in Z, msol, and RA, DEC,  MAG_r
IDS = list()
for m, r, d, rs, mag, i in zip(mass_survey, RA_survey, DEC_survey, Z_survey, MAG_r, np.arange(mass_survey.size)):
    if p.classify(m, r, d, rs, mag, p.lims) == True:
        IDS.append(i)

IDS = np.array(IDS)

RA = RA_survey[IDS]
DEC = DEC_survey[IDS]
Dist = Z_survey[IDS]*consts.c.to_value('km/s')/(p.h*100)
log_mass = np.log10(mass_survey[IDS])
MAG_r = MAG_r[IDS]

# Scale RA and DEC for the DBSCAN algorith to find outliers
sRA = RA - np.min(RA)
sDEC = DEC - np.min(DEC)
sRA = sRA/np.max(sRA)
sDEC = sDEC/np.max(sDEC)

print("Let's detect the outliers!")
X = np.vstack([sRA, sDEC]).T
# Do the outlier detection.. parameters carefully chosen
outlier_detection = DBSCAN(eps=0.05, metric='euclidean', min_samples=500, n_jobs=-1)
clusters = outlier_detection.fit_predict(X)

print("Done with outliers")

IDS_keep = np.where(clusters != -1)
IDS_rem = np.where(clusters == -1)

npixs_out = hp.ang2pix(p.nside, np.pi/2-np.deg2rad(DEC[IDS_rem]), np.deg2rad(RA[IDS_rem]))
npixs_in = hp.ang2pix(p.nside, np.pi/2-np.deg2rad(DEC[IDS_keep]), np.deg2rad(RA[IDS_keep]))

# Make plots to see how the outlier detection algorithm did
hp.mollview(np.zeros(12), rot=180)
hp.projscatter(np.pi/2-np.deg2rad(DEC[IDS_rem]), np.deg2rad(RA[IDS_rem]), s=1, c='red')
hp.projscatter(np.pi/2-np.deg2rad(DEC[IDS_keep]), np.deg2rad(RA[IDS_keep]), s=0.001, c='blue')
plt.savefig("../../Plots/Corrfunc/2_Mollview_remGals.png", dpi=180)

m = np.zeros(hp.nside2npix(p.nside))
m[npixs_in] = -1
m[npixs_out] = 1
hp.mollview(m, rot=180)
plt.savefig("../../Plots/Corrfunc/2_Mollview_remGalspixs.png", dpi=180)

plt.close()

# Grab the not-outliers galaxies
RA = RA[IDS_keep]
DEC = DEC[IDS_keep]
Dist = Dist[IDS_keep]
log_mass = log_mass[IDS_keep] 
MAG_r = MAG_r[IDS_keep]

pix_area = hp.nside2pixarea(p.nside, degrees=True)
Npix = np.unique(npixs_in).size
survey_area = pix_area*Npix
N = Dist.size
print("After applying cuts, there are {:d} galaxies over an area of {:.2f} deg^2".format(N, survey_area))

m = np.zeros(hp.nside2npix(p.nside))
pixs = hp.ang2pix(p.nside, np.pi/2-np.deg2rad(DEC), np.deg2rad(RA))
for p in pixs:
    m[p] = 1

# Make a mollview map of pixels over which random catalog will be distributed
hp.mollview(m, rot=[180, 0, 0])
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
pixs_list = np.save("../../Data/gpixs_list.npy", np.unique(npixs_in))

print("Done with cuts!")
