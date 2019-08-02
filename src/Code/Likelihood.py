#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import AbundanceMatching as amatch
import Corrfunc
import pickle
from pathos.pools import ProcessPool
from time import time
import Setup as p

class Posterior:
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
        self.nside = int(p.boxsize/p.subside)
        # Load observational correlation function
        obs_CF = p.load_pickle("/mnt/zfsusers/rstiskalek/Data/Obs_CF.p") 
        self.obs_wp = obs_CF["mean_wp"]
        self.obs_covmat = obs_CF["covmap_wp"]
        


    def __getAbundanceFunc(self, MFobj, mlim):
        """
        Unpack the MF, make some cut to eliminate unreliable data and return struct. array.
        """
        IDS = np.where(MFobj[:, 0]>mlim)
        # Abundances are not logged, whereas masses are logged
        af = amatch.AbundanceFunction(MFobj[:, 0][IDS], MFobj[:, 1][IDS], (mlim, 14),
                        faint_end_first=True)
        return af

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


    def abundance_match(self, alpha, scatter, Niter, ncores=1):
        """
        Does abundance matching with some alpha and scatter. Produces "niter" different catalogs due to
        random variations.
        """
        # Halo matching proxy
        plist = self.halos['vvir']*(self.halos['vmax']/self.halos['vvir'])**alpha
        # Calculate the number densities
        nd_halos = amatch.calc_number_densities(plist, p.boxsize)
        self.af.deconvolute(scatter, repeat=40)

        catalog = self.af.match(nd_halos)
        catalog_deconv = self.af.match(nd_halos, scatter, False)
        pool = ProcessPool(nodes=ncores)

        def add_scatter(i):
            cat_this = amatch.add_scatter(catalog_deconv, scatter)
            cat_this = amatch.rematch(cat_this, catalog, self.af._x_flipped)
            
            # Save the catalog.. Create a structured numpy array
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
            return cat_out
        res = pool.map(add_scatter, range(Niter))

        pool.close()
		pool.join()
		pool.clear()

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

    def loglikelihood(self, alpha, scatter):
        """

        """
        # OK. First generate Niter random catalogs to quantify random variations
        s = time()
        catalogs = self.abundance_match(alpha, scatter, Niter=25)
        print("Generated catalogs in {} seconds".format(time()-s))

        s = time()
        stochastic_covmat, stochastic_mean = self.stoch_covmat_mean(catalogs)
        print("Did stochastic statistics in {} seconds".format(time()-s))
        # Calculate jackknifed covariance matrix from the AM mock. Do this only for the first catalog
        s = time()
        __, jackknife_covmat = self.jackknife_sim(catalogs[0], plots=False)
        print("Did jackknifing in {} seconds".format(time()-s))

        # Sum up the covariance matrices
        covmat = stochastic_covmat + jackknife_covmat + self.obs_covmat
        
        # Get the log likelihood
        k = p.nbins
        det = np.linalg.det(covmat)
        inv = np.linalg.inv(covmat)
        diff = (stochastic_mean - self.obs_wp).reshape(k, 1)
        exponent = float(np.matmul(np.matmul(diff.T, covmat), diff))
        return -0.5*k*np.log(2*np.pi) - 0.5*np.log(det) - 0.5*exponent
    
    def logprior(self, alpha, scatter):
        """
        Pass
        """
        C1 = p.min_alpha < alpha < p.max_alpha
        C2 = p.min_scatter < scatter < p.max_scatter
        if C1 and C2:
            return np.log(p.max_alpha-p.min_alpha) + np.log(p.max_scatter-p.min_scatter)
        else:
            return -np.infty
            

def main():
    print("Entering main:")
    model = Posterior()
    print("Initiated the model.")
    print('ll', model.loglikelihood(0.5, 0.16))

if __name__ == '__main__':
    main()



