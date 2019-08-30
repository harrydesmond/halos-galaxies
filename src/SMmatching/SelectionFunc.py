#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from scipy.stats import binned_statistic, binned_statistic_2d
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import quad
from astropy.io import fits, ascii
import Setup as p

matched_cat = fits.open("../../BAM/a100springfull.fits")[1].data
logMH = matched_cat['logMH']
MS = matched_cat['SERSIC_MASS']
RA = matched_cat['RAdeg_HI']
DEC = matched_cat['DECdeg_HI']
Dist = matched_cat['Dist']
logW50 = matched_cat['W50']

h=0.688
H0 = 100*h

# Calculate the failure rate -> survival function
nbins = 20
bins_dist = np.linspace(np.min(Dist), np.max(Dist), nbins+1)
bins_logMH = np.linspace(logMH.min(), logMH.max(), nbins+1)

x_dist = p.bin_centers(bins_dist)
y_logMH = p.bin_centers(bins_logMH)
XX, YY = np.meshgrid(x_dist, y_logMH)

# Find the missing galaxies
IDS = np.where(np.isfinite(MS)==False)
remDist = Dist[IDS]
remHI = logMH[IDS]

stat =  binned_statistic_2d(x=remDist, y=remHI, values=remDist, statistic='count', bins=[bins_dist, bins_logMH])[0].T
stat_norm = binned_statistic_2d(x=Dist, y=logMH, values=Dist, statistic='count', bins=[bins_dist, bins_logMH])[0].T

def rate_interp(mass, dist, ZZ):
    xi = np.abs(y_logMH-mass).argmin()
    yi = np.abs(x_dist-dist).argmin()
    return ZZ[xi, yi]

def survival_rate(dist_arr, Mmin, Mmax, nsmooth, order):
    N = dist_arr.size
    failed_gals = np.zeros(N)
    total_gals = np.zeros(N)

    mass_arr = np.linspace(Mmin, Mmax, 200)
    for i in range(N):
        for m in mass_arr:
            failed_gals[i] += rate_interp(m, dist_arr[i], stat)
            total_gals[i] += rate_interp(m, dist_arr[i], stat_norm)

    survival_rate_coarse = 1-failed_gals/total_gals
    survival_rate_smooth = savgol_filter(survival_rate_coarse, nsmooth, order)
    survival_func = interp1d(dist_arr, survival_rate_smooth, kind='cubic', fill_value='extrapolate')
    return survival_rate_coarse, survival_rate_smooth, survival_func

# Calculate HI selection function
MFWascii = ascii.read("../../Data/HIdata/a100.180315.MWF.csv")
MFW = np.zeros(shape=(25, 18))
for i in range(25):
    for j in range(18):
        MFW[i, j] = MFWascii[i][j]

massbins = [6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8, 8.0, 8.2,
            8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0]
w50bins = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]

# Func to calculate m_HI,lim(d, w50)
def limit_logMH(d, w50):
    if w50 <= 2.5:
        logflux = 0.5*w50-1.207
    elif w50 > 2.5:
        logflux = w50-2.457
    return 5.372+logflux+2*np.log10(d)

def surv_dist_HI(dist, minmass):
    res = 0
    norm = 0
    mlow_ind = np.abs(minmass-np.array(massbins)).argmin()

    for i in range(len(w50bins)-1):
        dw50 = w50bins[i+1]-w50bins[i]
        w50 = (w50bins[i+1]+w50bins[i])/2

        mlim = limit_logMH(dist, w50)

        mlim_ind = np.abs(mlim-np.array(massbins)).argmin()

        if mlim < minmass:
            xi = mlow_ind
        else:
            xi = mlim_ind

        for j in range(xi, len(massbins)-1):
            dm = massbins[j+1]-massbins[j]
            res += MFW[j, i]*dm*dw50

        for j in range(mlow_ind, len(massbins)-1):
            dm = massbins[j+1]-massbins[j]
            norm += MFW[j, i]*dm*dw50

    return res/norm

def survival_func_HI(dist_arr, mlim, points):
    y = np.array([surv_dist_HI(i, mlim) for i in dist_arr])
    # Apply filter
    yhat = savgol_filter(y, points, 1)
    # Interpolate the filter
    f = interp1d(dist_arr, yhat)
    return f

# Calculate RFI survival

# Sun's velocity vector
vsun = 369
RAsun= (11+11/60+57/60**2)*np.pi/12
DECsun = np.deg2rad(7.22)

def heliovelocity(pars):
    RAcmb, DECcmb, distcmb = pars
    cos_theta = np.cos(DECcmb)*np.cos(DECsun)*np.cos(RAcmb-RAsun)+np.sin(DECcmb)*np.sin(DECsun)
    return H0*distcmb-vsun*cos_theta

dataRFI = ascii.read("../../Data/HIdata/rfi_frac.ascii")

rfi_hvel = np.array(dataRFI['Vhelio'])
rfi = np.array(dataRFI['fRFI'])

survival_func_rfi = interp1d(rfi_hvel, rfi, bounds_error=False, fill_value='extrapolate')

# Calculate the SDSS survival function

dataLF = np.loadtxt("../../Data/SMmatching/LF_Ser.dat")

xLF = dataLF[:, 0]
yLF = dataLF[:, 1]

# Interpolate the galaxy luminosity function
nlum = interp1d(xLF, yLF, kind='cubic')
t = np.linspace(xLF.min(), xLF.max(), 1000)

# Extrapolate the faint and bright end
def bright_end_func(x, a, b, c, d):
    return -np.exp(a*x+b) + c*x + d

def faint_end_func(x, a, b):
    return a*x+b

bright_end_coef = curve_fit(bright_end_func, xLF[:27], yLF[:27], [[1, 0, 0, 0]], maxfev=100000)[0]
s = slice(-5, None)
faint_end_coef = curve_fit(faint_end_func, xLF[s], yLF[s], [0, 0], maxfev=100000)[0]


tbright = np.linspace(-24.7, -30, 250)
a, b, c, d = bright_end_coef
ybright = [bright_end_func(i, a, b, c, d) for i in tbright]

tfaint=np.linspace(-17.6, -12, 250)
a, b = faint_end_coef
yfaint = [faint_end_func(i, a, b) for i in tfaint]

y = nlum(t)

# Merge the arrays
xx = np.hstack([tfaint, t, tbright])
yy = np.hstack([yfaint, y, ybright])

IDS = np.argsort(xx)
xx = xx[IDS]
yy = yy[IDS]

IDS = np.unique(xx, return_index=True)[1]
xx = xx[IDS]
yy = yy[IDS]
# Extrapolated luminosity function
nlum = interp1d(xx, 10**yy, kind='cubic')

# Unpack all the values
ra = matched_cat['RAdeg_HI']
dec = matched_cat['DECdeg_HI']
dist = matched_cat['Dist']
logMH = matched_cat['logMH']
MS = matched_cat['SERSIC_MASS']
appmag = matched_cat['petroMag_r']

# Take only the ones with finite SM
IDS = np.where(np.isfinite(MS))
ra = ra[IDS]
dec = dec[IDS]
dist = dist[IDS]
logMH = logMH[IDS]
logMS = np.log10(MS[IDS])
appmag = appmag[IDS]

def redshift2distance(z):
    omega_m = 0.295
    omega_lambda = 0.705
    omega_k = 0.0
    H0 = 68.8
    Dh = scipy.constants.c*1e-3/(H0)
    num = lambda z_prime : 1/np.sqrt(omega_m*(1+z_prime)**3+omega_k*(1+z_prime)**2+omega_lambda)
    quadnum = quad(num, 0, z)[0]
    comoving_distance = Dh*quadnum
    return comoving_distance

xx_redshift = np.linspace(0, 1, 10000)
yy_comoving = np.array([redshift2distance(i) for i in xx_redshift])

distance2redshift = interp1d(yy_comoving, xx_redshift)

def func_absmag(appmag, comoving_dist):
    # Return absolute magnitude for a given apparent magnitude and comoving distance
    z = distance2redshift(comoving_dist)
    luminosity_distance = (1+z)*comoving_dist
    return appmag-25-5*np.log10(luminosity_distance)

absmag = list()
for i, j in zip(appmag, dist):
    absmag.append(func_absmag(i, j))
absmag = np.array(absmag)

faintest_absmag = np.max(absmag)
brightest_absmag = np.min(absmag)

def survival_func_SM(dist_arr, faintest_absmag):
    norm = quad(nlum, faintest_absmag, brightest_absmag)[0]
    
    y = list()
    for d in dist_arr:
        sdsslim = func_absmag(17.6, d)
        # If SDSS lim is less bright than faintest sdss object return 1.0
        if sdsslim > faintest_absmag:
            y.append(1.0)
        else:
            integ = quad(nlum, sdsslim, brightest_absmag, limit=100, epsabs=1.49e-5)
            if integ[1] > 1e-3:
                print("Hmm, having problems")
            y.append(integ[0]/norm)
    
    return interp1d(dist_arr, y, kind='cubic')

