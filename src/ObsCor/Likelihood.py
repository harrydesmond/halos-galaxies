#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import AbundanceMatching as amatch
from time import time
import Setup as p
#from astropy import constants as const, units as u


class MasterEquation:
    """
    Add a fancy description
    """
    def __init__(self):
        # Load and unpack the MF
        MFobj = np.loadtxt("../../BAM/SMF_bin_abundance.dat")
        self.af = self.__getAbundanceFunc(MFobj, mlim=6.8)
        # Load in the list of halos (this list assumed to be already edited..)
        self.halos = np.load("../../Data/halos_list.npy")
        # Define cosmology. Not so sure about little h :(
        cosmology = {u'h': 0.688, u'omega_lambda_0': 0.705, u'omega_k_0': 0.0, u'omega_M_0': 0.295}

    def __getAbundanceFunc(self, MFobj, mlim):
        """
        Unpack the MF, make some cut to eliminate unreliable data and return struct. array.
        """
        IDS = np.where(MFobj[:, 0]>mlim)
        # Abundances are not logged, whereas masses are logged
        af = amatch.AbundanceFunction(MFobj[:, 0][IDS], MFobj[:, 1][IDS], (5.0, 13.0),
                        faint_end_first=True)
        return af
    
    def abundance_match(self, alpha, scatter, Niter=5):
        """
        Some description
        """
        # Halo matching proxy
        plist = self.halos['vvir']*(self.halos['vmax']/self.halos['vvir'])**alpha
        # Calculate the number densities
        nd_halos = amatch.calc_number_densities(plist, p.boxsize)
        names = ['mvir', 'conc', 'galmass', 'x', 'y', 'z']    
        
        cat_out = list()
        for n in range(Niter):
            # What is the appropriate choice for repeat?
            # Maybe this choice of repeat can make this faster
            # Add the moment it takes about 37 seconds for each AM
            # This takes about 37 seconds for each iteration at the moment..
            self.af.deconvolute(scatter, repeat=20)
            # Do abundance matching by matching the number densities. Here 0 scatter
            cat = self.af.match(nd_halos, scatter)
            # Find masks to eliminate what we don't like
            mask = (~np.isnan(cat)) & (cat>7.)
            mvir = self.halos['mvir'][mask]
            N = mvir.size
            # Create a structured array
            catalog = np.zeros(N, dtype={
                         'names':('mvir', 'conc', 'galmass', 'x', 'y', 'z'),
                         'formats':('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
                                 })
            data = [self.halos['mvir'][mask], self.halos['mvir'][mask], cat[mask],
                    self.halos['x'][mask], self.halos['y'][mask], self.halos['z'][mask]] 
            for name, d in zip(names, data):
                catalog[name] = d

            cat_out.append(catalog)
            print("Done with {}/{}".format(n+1, Niter))

        return cat_out


    def loglikelihood(self, alpha):
        pass






def main():
    print("Entering main:")
    model = MasterEquation()
    print("Initiated the model.")
    
    scatter = 0.16
    alpha = 0.0
    start = time()
    a = model.abundance_match(alpha, scatter)
    print("Did 40 iters in {} seconds".format(time()-start))



if __name__ == '__main__':
    main()



