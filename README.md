# The relation between galaxies and dark matter halos

The aim of this work is to perform baryonics mass based abundance matching. This code contains (in no particular order) tools to calculate the projected correlation function from galaxy surveys and compare that to an abundance matched catalog. For the observational data we use a matched catalog from ALFALFA and NSA to obtain baryonic mass estimates and DarkSky N-body simulation.

## The files work in the following order (for BM abundance matching)
  1. Generate random sample which has the same spatial coverage as the galaxy survey and apply selection effects to it in BM_CatalogPreparation.ipynb
  2. Generate the observed correlation function using previously calculated catalogs in CFobs.py
  3. Setup appropriate boundaries for the parameter grid search in Setup.py
  4. Find the jackknifed + stochastic covariance matrices + correlation from abundance matching by running Jackknife_covmat_generate.py and StochCovmat_generate.py (don't forget to split the jobs onto an appropriate number of cores)
  5. Grid search the likelihood in GridSearch.ipynb and make some sweet corner plots!

  
