#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import scipy.interpolate
import AbundanceMatching as amatch
import Corrfunc
from time import time
import Setup as p

class Model:
    """
    A likelihood and prior for abundance matching parameters fitting to the given precomputed projected
    two-point correlation function.
    
    Note:
        The parameters are rather inflexible right now. Changing from stellar mass abundance matching to
        baryonic mass etc. requires changing some codes within this and precomputing a plenty of things.
        A well needed description is about to be written in README.
    """
    def __init__(self, logMBlim, perccut, generator):
        # Load and unpack the MF
        self.logMBlim = logMBlim
        self.perccut = perccut
        self.boxsize = 400.0 #Mpc/h
        self.MFobj = np.loadtxt("../../BAM/BMF_bin_abundance.dat")
        self.af = self.__getAbundanceFunc(self.MFobj)
        # Load in the list of halos (this list assumed to be already edited..)
        self.halos = self.get_halos(np.load("../../Data/BMmatching/halos_list.npy"))
        self.rp_bins =  p.bins # Mpc/h
        self.bins_arr = np.arange(p.nbins)
        self.nside = 16
        # Load observational correlation function
        obs_CF = p.load_pickle("../../Data/BMmatching/Obs_CF_BMcut_{}_.p".format(self.perccut)) 
        self.obs_wp = obs_CF["mean_wp"]
        self.obs_covmat = obs_CF["covmap_wp"]
        if generator == False:
            # Load the precomputed covariance matrices
            self.covmat_interp, self.alphas_unique, self.scatters_unique = self.interp_covmat()
            self.CFmean_interp = self.interp_mean()


    def __getAbundanceFunc(self, MFobj):
        """
        Returns the abundance matching function object.

        Args:
            MFobj : precomputed mass function. Assumes both columns are logged
        """
        IDS = np.where(MFobj[:, 0] > 6.8)
        # Abundances are not logged, whereas masses are logged
        af = amatch.AbundanceFunction(MFobj[:, 0][IDS], MFobj[:, 1][IDS], (5, 13), faint_end_first=True)
        return af
    
    def interp_covmat(self):
        """
        Loads the precomputed values of covariance matrix on a grid and returns the interpolation object.
        """
        data_jack = p.load_pickle("../../Data/BMmatching/Train_jackknife_covmats_{}_.p".format(self.perccut))
        data_stoch = p.load_pickle("../../Data/BMmatching/Train_stoch_covmats_{}_.p".format(self.perccut))
        alphas_unique = np.unique(data_jack['alpha'])
        scatters_unique = np.unique(data_jack['scatter'])
        z_covmat = data_jack['covmat'] + data_stoch['covmat']
        return z_covmat, alphas_unique, scatters_unique


    def interp_mean(self):
        """
        Load the precomputed values of mean correlation function from stochastic realisations. Return the
        interpolation object.
        """
        data = p.load_pickle("../../Data/BMmatching/Train_stoch_covmats_{}_.p".format(self.perccut))
        wp = data['wp']
        return wp


    def get_halos(self, halos_object):
        """
        Bins the halos, creates a new struct array
        which also contains the bin in which each halo located. Useful for jackknifing.

        Args:
            halos_object : numpy structures array containing the halo properties
        """
        # Calculate the box position in a subvolume
        edges = np.arange(0, self.boxsize+p.subside, p.subside)
        # Figure out into which boxes galaxies belong
        lx = np.digitize(halos_object['x'], edges)-1
        ly = np.digitize(halos_object['y'], edges)-1
        nboxes = edges.size-1
        gbins = ly*nboxes+lx
            
        N = halos_object['mvir'].size    
        halos_catalog = np.zeros(N, dtype={'names':('mvir', 'x', 'y', 'z', 'vmax', 'vvir', 'gbins'),
                              'formats':('float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64')})
        names = ['mvir', 'x', 'y', 'z', 'vmax', 'vvir', 'gbins']
        data = [halos_object['mvir'], halos_object['x'], halos_object['y'], halos_object['z'],
                halos_object['vmax'], halos_object['vvir'], gbins]
        for name, d in zip(names, data):
            halos_catalog[name] = d

        return halos_catalog

    def get_covmat_mean(self, alpha, scatter):
        """
        Returns the covariance matrix from the interpolation object.

        NEEDS UPDATE  doc
        Args:
            alpha : (float) halo matching proxy parameter
            scatter : (float) halo matching proxy parameter
        """
        xi = np.abs(self.alphas_unique-alpha).argmin()
        yi = np.abs(self.scatters_unique-scatter).argmin()
        return self.covmat_interp[yi, xi], self.CFmean_interp[yi, xi]


    def abundance_match(self, alpha, scatter, Niter, repeat=20):
        """
        Returns a list of abundance matching catalogs of given alpha and scatter.

        Args:
            alpha : (float) halo matching proxy parameter
            scatter : (float) halo matching proxy parameter
            min_galmass: (float) minimum galactic mass in the catalog in log10
            Niter : (int) number of catalogs to be calculated
        
        Note:
            This function is single core only. Might bottleneck some calculations
        """
        # Halo matching proxy
        plist = self.halos['vvir']*(self.halos['vmax']/self.halos['vvir'])**alpha
        # Calculate the number densities
        nd_halos = amatch.calc_number_densities(plist, self.boxsize)

        res = list()
        for __ in range(Niter):
            self.af.deconvolute(scatter, repeat)
            cat_this = self.af.match(nd_halos, scatter)
            # Eliminate NaNs and galaxies with mass lower cut
            mask = (~np.isnan(cat_this)) & (cat_this>self.logMBlim)
            # TO DO:
            #       - when doing this cut include a parameter that will exclude some low mass halos
            N = np.where(mask == True)[0].size
            cat_out = np.zeros(N, dtype={'names':('mvir', 'cat', 'x', 'y', 'z', 'gbins'),
                              'formats':('float64', 'float64', 'float64', 'float64', 'float64', 'int64')})
            cat_out['mvir'] = self.halos['mvir'][mask]
            cat_out['cat'] = cat_this[mask]
            cat_out['x'] = self.halos['x'][mask]
            cat_out['y'] = self.halos['y'][mask]
            cat_out['z'] = self.halos['z'][mask]
            cat_out['gbins'] = self.halos['gbins'][mask]
            
            res.append(cat_out)

        return res


    def jackknife_sim(self, catalog, nthreads=p.nthreads):
        """
        Jackknifes the simulation box and returns the mean projected correlation function
        and the covariance matrix. Does this on a single catalog.

        Args:
            catalog : abundance matching catalog
            nthreads : (int) nthreads used for corrfunc calculation
        """
        Nsub = self.nside**2
        wp_out = list()
        for i in range(Nsub):
            IDS = np.where(catalog['gbins'] != i)
            XX = catalog['x'][IDS]
            YY = catalog['y'][IDS]
            ZZ = catalog['z'][IDS]
            wp = Corrfunc.theory.wp(boxsize=self.boxsize, pimax=p.pimax, nthreads=nthreads, binfile=self.rp_bins, X=XX, Y=YY, Z=ZZ)
            wp_out.append(wp['wp'])
        wp_out = np.array(wp_out)
        
        mean_wp = np.mean(wp_out, axis=0)
        cov_matrix = np.zeros((p.nbins, p.nbins))

        for i in range(p.nbins):
            for j in range(p.nbins):
                for k in range(Nsub):
                    cov_matrix[i, j] += (wp_out[k, i]-mean_wp[i])*(wp_out[k, j]-mean_wp[j])

        cov_matrix = cov_matrix*(Nsub-1)/Nsub

        return mean_wp, cov_matrix

    def stoch_covmat_mean(self, catalogs, nthreads=p.nthreads):
        """
        Calculates the mean and covariance matrix for a list of catalogs. Assumes catalogs are independent for
        1/N normalisation. Returns the mean CF and the covariance matrix

        Args:
            catalogs : (list) list of abundance matched catalogs
            nthreads : (int) number of cores to be used for CF calculation
        """
        # First calculate the wp for each catalog
        wps = list()
        for catalog in catalogs:
            XX = catalog['x']
            YY = catalog['y']
            ZZ = catalog['z']
            wp = Corrfunc.theory.wp(boxsize=self.boxsize, pimax=p.pimax, nthreads=nthreads, binfile=self.rp_bins,
                                    X=XX, Y=YY, Z=ZZ)
            wps.append(wp['wp'])
        
        wps = np.array(wps)
        wp_mean = np.mean(wps, axis=0)
        
        # Calculate the covariance matrix
        cov_matrix = np.zeros((p.nbins, p.nbins))
        for i in range(p.nbins):
            for j in range(p.nbins):
                for k in range(len(catalogs)):
                    cov_matrix[i, j] +=  (wps[k, i]-wp_mean[i])*(wps[k, j]-wp_mean[j])
        
        cov_matrix = cov_matrix/len(catalogs)
        return cov_matrix, wp_mean

    def loglikelihood(self, theta):
        """
        The natural logarithm of the likelihood. Model assumes Gaussian. Note: ignores
        normalisation.

        Args:
            theta (tuple): individual parameter values (alpha, scatter)
        """
        # Unpack the parameters
        alpha, scatter = theta
        # Get the interpolated covariance matrix and mean CF
        simcovmat, wp_mean = self.get_covmat_mean(alpha, scatter)
        # Sum all covariance matrices
        covmat = simcovmat + self.obs_covmat
        # Determinant and inverse of the covariance matrix
        det = np.linalg.det(covmat)
        inv = np.linalg.inv(covmat)
        # Difference between parameter and observed correlation function
        diff = (wp_mean - self.obs_wp).reshape(p.nbins, 1)
        # Do the matrix product for the exponent part of the multivariate Gaussian likelihood
        exponent = float(np.matmul(np.matmul(diff.T, inv), diff))
        return -0.5*(np.log(det) + exponent)
    
    def logprior(self, theta):
        """
        The natural logarithm of the prior probability. Note: ignores normalisation.
        Hence returns 0 for uniform distribution.

        Args:
            theta (tuple): individual parameter values (alpha, scatter)

        Note:
            For emcee the uniform prior is neglected.
        """
        # Unpack the values
        alpha, scatter = theta
        # Check if withing range. Assuming uniform prior
        cond1 = p.min_alpha < alpha < p.max_alpha
        cond2 = p.min_scatter < scatter < p.max_scatter
        if cond1 and cond2:
            return 0.0 
        else:
            return -np.inf



def main():
    print("Entering main:")
    model = Model(9.0)
    print("Initiated the model.")
    start = time()
    ll = model.loglikelihood(0.5, 0.16)
    lp = model.logprior(0.5, 0.16)
    print(ll)
    print("Sampled the likehood in {}.".format(time()-start))



if __name__ == '__main__':
    main()
