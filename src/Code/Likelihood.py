#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate
import AbundanceMatching as amatch
import Corrfunc
import pickle
from pathos.pools import ProcessPool
from time import time
import Setup as p

class Model:
    """
    A likelihood and prior for abundance matching parameters fitting to SDSS data.
    """
    def __init__(self):
        # Load and unpack the MF
        MFobj = np.loadtxt("/mnt/zfsusers/rstiskalek/BAM/SMF_bin_abundance.dat")
        self.af = self.__getAbundanceFunc(MFobj, mlim=7.5)
        # Load in the list of halos (this list assumed to be already edited..)
        self.halos = self.get_halos(np.load("/mnt/zfsusers/rstiskalek/Data/halos_list.npy"), 7.5)
        self.rp_bins = np.logspace(np.log10(p.min_rp), np.log10(p.max_rp), p.nbins+1)
        self.bins_arr = np.arange(p.nbins)
        self.nside = int(p.boxsize/p.subside)
        # Load observational correlation function
        obs_CF = p.load_pickle("/mnt/zfsusers/rstiskalek/Data/Obs_CF.p") 
        self.obs_wp = obs_CF["mean_wp"]
        self.obs_covmat = obs_CF["covmap_wp"]
        # Load the precomputed covariance matrices
        self.covmat_interp = self.interp_covmat()



    def __getAbundanceFunc(self, MFobj, mlim):
        """
        Unpack the MF, make some cut to eliminate unreliable data and return struct. array.
        """
        IDS = np.where(MFobj[:, 0]>mlim)
        # Abundances are not logged, whereas masses are logged
        af = amatch.AbundanceFunction(MFobj[:, 0][IDS], MFobj[:, 1][IDS], (mlim, 14),
                        faint_end_first=True)
        return af
    
    def interp_covmat(self, method='nearest'):
        """
        Load the precomputed values of covariance matrix on a grid and creates and interpolation object
        """
        data_jack = p.load_pickle("../../Data/Train_jackknife_covmats.p")
        #data_stoch = p.load_pickle("../../Data/Train_stoch_covmats.p")
        XX = data_jack['alpha']
        YY = data_jack['scatter']
        z_covmat = data_jack['covmat']
        f = scipy.interpolate.RegularGridInterpolator(points=(np.unique(YY), np.unique(XX), self.bins_arr, self.bins_arr),
                                                      values=z_covmat, method='nearest')
        return f


    def get_halos(self, halos_object, cutoff, subside=p.subside):
        """
        Bin the halos and create a new struct array
        """
        # Calculate the box position in a subvolume
        edges = np.arange(0, p.boxsize+subside, subside)
        # Figure out into which boxes galaxies belong
        lx = np.digitize(halos_object['x'], edges)-1
        ly = np.digitize(halos_object['y'], edges)-1
        nboxes = edges.size-1
        gbins = ly*nboxes+lx

        IDS = np.where(np.log10(halos_object['mvir']) > cutoff)
        N = halos_object['mvir'][IDS].size    
        halos_catalog = np.zeros(N, dtype={'names':('mvir', 'x', 'y', 'z', 'vmax', 'vvir', 'gbins'),
                              'formats':('float64', 'float64', 'float64', 'float64', 'float64', 'float64', 'int64')})
        names = ['mvir', 'x', 'y', 'z', 'vmax', 'vvir', 'gbins']
        data = [halos_object['mvir'][IDS], halos_object['x'][IDS], halos_object['y'][IDS], halos_object['z'][IDS],
                halos_object['vmax'][IDS], halos_object['vvir'][IDS], gbins[IDS]]
        for name, d in zip(names, data):
            halos_catalog[name] = d

        return halos_catalog

    def comp_simcovmat(self, alpha, scatter):
        """
        Computes the covariance matrix given the interpolation object
        """
        covmat = np.zeros(shape=(30, 30))
        for i in range(30):
            for j in range(30):
                covmat[i, j] = self.covmat_interp((scatter, alpha, i, j))
        return covmat


    def abundance_match(self, alpha, scatter, Niter):
        """
        Does abundance matching with some alpha and scatter. Produces "niter" different catalogs due to
        random variations.
        """
        # Halo matching proxy
        plist = self.halos['vvir']*(self.halos['vmax']/self.halos['vvir'])**alpha
        # Calculate the number densities
        nd_halos = amatch.calc_number_densities(plist, p.boxsize)

        res = list()
        for __ in range(Niter):
            self.af.deconvolute(scatter, 40)
            cat_this = self.af.match(nd_halos, scatter)
            # Eliminate NaNs and galaxies with mass lower cut
            mask = (~np.isnan(cat_this)) & (cat_this>9.8)
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


    def jackknife_sim(self, catalog, nthreads=p.nthreads, plots=False):
        """
        Perform jackknifing on a single abundance matching realisation.
        """
        Nsub = self.nside**2
        wp_out = list()
        for i in range(Nsub):
            IDS = np.where(catalog['gbins'] != i)
            XX = catalog['x'][IDS]
            YY = catalog['y'][IDS]
            ZZ = catalog['z'][IDS]
            wp = Corrfunc.theory.wp(boxsize=p.boxsize, pimax=p.pimax, nthreads=nthreads, binfile=self.rp_bins, X=XX, Y=YY, Z=ZZ)
            wp_out.append(wp['wp'])
        
        wp_out = np.array(wp_out)
        
        mean_wp = np.mean(wp_out, axis=0)
        cov_matrix = np.zeros((p.nbins, p.nbins))

        for i in range(p.nbins):
            for j in range(p.nbins):
                for k in range(Nsub):
                    cov_matrix[i, j] += (wp_out[k, i]-mean_wp[i])*(wp_out[k, j]-mean_wp[j])

        cov_matrix = cov_matrix*(Nsub-1)/Nsub

        if plots ==True:
            cbins = p.bin_centers(self.rp_bins)
            std = np.sqrt(np.diagonal(cov_matrix))

            fig = plt.figure()
            ax = plt.axes()
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.errorbar(cbins, mean_wp, yerr=std, label="Simulated CF")
            ax.plot(p.xr, p.yr, label='Data from Reddick', marker='o')
            ax.set_xlabel(r'$r_p$')
            ax.set_ylabel(r'$w_p$')
            ax.set_ylim(bottom=10**(-1), top=10**(4))
            ax.legend()
            plt.tight_layout()   
            plt.savefig("/mnt/zfsusers/rstiskalek/Plots/Corrfunc/4_CFcomparSIM.png", dpi=180)
            plt.close()


        return mean_wp, cov_matrix

    def stoch_covmat_mean(self, catalogs, nthreads = p.nthreads, plots=False):
        """
        Returns the stochastic covariance matrix and mean wp due to variations in random catalogs.
        """
        # First calculate the wp for each catalog
        wps = list()
        for catalog in catalogs:
            XX = catalog['x']
            YY = catalog['y']
            ZZ = catalog['z']
            wp = Corrfunc.theory.wp(boxsize=p.boxsize, pimax=p.pimax, nthreads=nthreads, binfile=self.rp_bins,
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
        alpha, scatter = theta
        simcovmat = self.comp_simcovmat(alpha, scatter)
        covmat = simcovmat + self.obs_covmat

        catalog = self.abundance_match(alpha, scatter, Niter=1)
        __, wp_mean = self.stoch_covmat_mean(catalog)
        
        # Get the log likelihood
        det = np.linalg.det(covmat)
        inv = np.linalg.inv(covmat)
        diff = (wp_mean - self.obs_wp).reshape(p.nbins, 1)
        exponent = float(np.matmul(np.matmul(diff.T, inv), diff))
        return -0.5*(np.log(det) + exponent)
    
    def logprior(self, theta):
        """
        The natural logarithm of the prior probability. Note: ignores normalisation.
        Hence returns 0 for uniform distribution.

        Args:
            theta (tuple): individual parameter values (alpha, scatter)
        """
        # Unpack the values
        alpha, scatter = theta
        # Check if withing range
        cond1 = p.min_alpha < alpha < p.max_alpha
        cond2 = p.min_scatter < scatter < p.max_scatter
        if cond1 and cond2:
            return 0.0 
        else:
            return -np.inf



def main():
    print("Entering main:")
    model = Model()
    print("Initiated the model.")
    start = time()
    ll = model.loglikelihood(0.5, 0.16)
    lp = model.logprior(0.5, 0.16)
    print(ll)
    print("Sampled the likehood in {}.".format(time()-start))



if __name__ == '__main__':
    main()



