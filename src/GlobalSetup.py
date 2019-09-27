#!/usr/bin/env python3
# coding: utf-8

boxsize = 400
nside = 25
subside = 25
nbins = 15
min_rp = 0.1
max_rp = 30
bins = np.logspace(np.log10(min_rp), np.log10(max_rp), nbins + 1)
pimax = 40

halos_file = ....
