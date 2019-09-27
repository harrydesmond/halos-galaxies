#!/usr/bin/env python3
# coding: utf-8
import os
from Likelihood import Model
from BayesSampler import Sampler

MFfile = "../../BAM/SMF_bin_abundance.dat"
logMlim = 9.8
name
new_model
halos = "lal"
obsCF = "../../Data/ ll"

boxsize = 400
nside = 25
subside = 25
nbins = 15
min_rp = 0.1
max_rp = 30
bins = np.logspace(np.log10(min_rp), np.log10(max_rp), nbins + 1)
pimax = 40

# Make a choice if want to start generating a new model
if new_model:
    # If dir doesnt exist make a new one!
    clean_directory = "rm ../../Data/{}/*"
    os.system(clean_directory)
    # model draw initial samples
else:
    # model load old samples
    

pbounds = .... 
parameters = ....

model = model(....)


