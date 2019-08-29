#!/usr/bin/env python
# coding: utf-8 
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cosmolopy import density
from astropy import constants as const, units as u
import Setup as p

def vvir_from_cosmology(halos, cosmology):
    """
    Computes the peak virial velocity from cosmology, to be used in computing the halo proxy.
    """
    z_mpeak = (1.0-halos['mpeak_scale'])/halos['mpeak_scale']
    OmegaM_mpeak = density.omega_M_z(z_mpeak, **cosmology)
    Delta_vir = (18.0*np.pi**2+82*(OmegaM_mpeak-1.0)-39.0*(OmegaM_mpeak-1.0)**2)/OmegaM_mpeak
    rho_crit, rho_mean = density.cosmo_densities(**cosmology)*u.Msun/(u.Mpc**3)
    Rvir_mpeak = (3.0*halos['mpeak']*u.Msun/(4.0*np.pi*rho_crit*Delta_vir))**(1.0/3.0)
    vvir_mpeak = np.sqrt(const.G*halos['mpeak']*u.astrophys.solMass/Rvir_mpeak).to(u.km/u.s)
    return vvir_mpeak.value

# Check about these annoying units of h! 
# Cosmological parameters
cosmology = {u'h': 0.688, u'omega_lambda_0': 0.705, u'omega_k_0': 0.0, u'omega_M_0': 0.295}
# Load in the halos data file
halos = np.load("../../BAM/hlist_1.00000.npy")

print("Finished loading things")
# And grab some stuff
vvir = vvir_from_cosmology(halos, cosmology)
vmax = halos['Vmax@Mpeak']
mvir = halos['mvir']/p.h # Put virial mass into physical Msun units
conc = (halos['rvir']/halos['rs_klypin'])
x_pos = halos['x']
y_pos = halos['y']
z_pos = halos['z']
pid = halos['pid']

print("Finished computing things")

# Save these into an npy data files
N = x_pos.size

halos_catalog = np.zeros(N, dtype={'names':('mvir', 'conc', 'x', 'y', 'z', 'vmax', 'vvir', 'pid'),
                      'formats':('float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64')})
names = ['mvir', 'conc', 'x', 'y', 'z', 'vmax', 'vvir', 'pid']
data = [mvir, conc, x_pos, y_pos, z_pos, vmax, vvir, pid]
for name, d in zip(names, data):
    halos_catalog[name] = d

np.save("../../Data/SMmatching/halos_list.npy", halos_catalog)

print("Finished with everything")
