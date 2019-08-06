#!/usr/bin/env python3
# coding: utf-8 
import numpy as np
import emcee
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import corner

import Setup as p
from Likelihood import Model

# Parse the input arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--Nburnin", dest="Nburnin", help="Number of MCMC burnin",
                    type=int, default=2500)
parser.add_argument("--Nsamples", dest="Nsamples", help="Number of MCMC samples",
                    type=int, default=1000)
parser.add_argument("--threads", dest="threads", help="Number of threads",
                    type=int, default=4)
parser.add_argument("--resume", dest="resume", help="True if resume previous run",
                    type=bool, default=None)
args = parser.parse_args()

Nens = p.Nens               # Number of ensemble points
Nburnin = args.Nburnin      # Number of burnin samples
Nsamples = args.Nsamples    # Number of posterior samples
threads = args.threads      # Number of threads
resume = args.resume        # Continue? bool

if resume == None:
    print("Resume argument not entered. Exiting")
    sys.exit()

# Initiate the model
model = Model()
print("Initiated the model. Starting!")

# Define log posterior
def logposterior(theta):
    lp = model.logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + model.loglikelihood(theta)

# Get initial positions. If resuming from previous run load the last positions
if resume == True:
    inisamples = p.load_pickle("../../Data/MCMC/Last_step.p")
else:
    alpha_ini = np.random.uniform(p.min_alpha, p.max_alpha, Nens)
    scatter_ini = np.random.uniform(p.min_scatter, p.max_scatter, Nens)
    inisamples = np.array([alpha_ini, scatter_ini]).T

# Number of dimensions (i.e. number of parameters)
ndims = inisamples.shape[1]

# Run the sampler 
sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, threads=threads)
sampler.run_mcmc(inisamples, Nsamples+Nburnin)

# Save the sampler (might be useful to look at it later)
p.dump_pickle(sampler, "../../Data/MCMC/Sampler.p")
# Get the posterior samples
postsamples = sampler.chain[:, Nburnin:, :].reshape((-1, ndims))


print('Number of posterior samples is {}'.format(postsamples.shape[0]))
print('Acceptance fraction is {}'.format(sampler.acceptance_fraction))

# Make some plots
fig = corner.corner(postsamples, labels=[r"$\alpha$", r"$\sigma$"])
fig.savefig("../../Plots/MCMC/Corner.png", dpi=180)

probchain = sampler.lnprobability
plt.figure()
for i in range(probchain.shape[0]):
    plt.plot(probchain[i, Nburnin:])
plt.savefig("../../Plots/MCMC/ProbChain.png", dpi=180)
plt.close()


