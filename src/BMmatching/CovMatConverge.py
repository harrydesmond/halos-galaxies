#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Setup as p
import Likelihood

model = Likelihood.Model()

alphas = [0.0, 0.5, 1.0]
scatters = [0.0, 0.15, 0.30]
Niter = 40

def frob_dist(cat0, cat1):
    return np.mean(np.sqrt(np.square(cat1-cat0)))

nthreads = 6

for alpha in alphas:
    for scatter in scatters:
        x = list()
        covmatdist = list()
        wpdist = list()
        catalogs = model.abundance_match(alpha, scatter, Niter)

        for i in range(2, Niter-1):
            covmat0, wp0 = model.stoch_covmat_mean(catalogs[:i], nthreads)
            covmatf, wpf = model.stoch_covmat_mean(catalogs[:i+1], nthreads)
            x.append(i+1)
            covmatdist.append(frob_dist(covmatf, covmat0))
            wpdist.append(frob_dist(wpf, wp0))
        
        
        # Make the plot
        plt.figure(figsize=(15, 10))
        
        plt.subplot(221)
        plt.plot(x, covmatdist, marker='o')
        plt.title(r"Variations of stoch. $C_{{ij}}$ for $\alpha={}, \sigma={}$".format(alpha, scatter))
        plt.ylabel(r"Mean Frob dist$(N_{cats}, N_{cats}-1)$")
        plt.xlabel(r"$N$ catalogs")
        plt.grid(True)
        
        plt.subplot(222)
        plt.plot(x, wpdist, marker='o')
        plt.title(r"Variations of stoch. mean $\widehat{{w}}_p$ for $\alpha={}, \sigma={}$".format(alpha, scatter))
        plt.xlabel(r"$N$ catalogs")
        plt.ylabel(r"Mean Frob dist$(N_{cats}, N_{cats}-1)$")
        plt.grid(True)
        
        plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
        plt.savefig("../../Plots/CMconv/Stoch_alpha{}_scatter{}_.png".format(alpha, scatter), dpi=180)
        plt.close()
print("Finished!")    
