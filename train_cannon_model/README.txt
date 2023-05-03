download_RVS_spectra.py	: queries all stars in ../process_GALAH_data/GALAH_data_tables/GALAH_stars_filtered.csv in Gaia, downloads RVS spectra and saves them to ./flux_data.csv and ./flux_err_data.csv

train_cannon_model.py : trains the cannon model

flux_data.csv : contains flux arrays for every star in the training + test sample, labeled by Gaia designation
flux_err_data.csv : contains flux error arrays for every star in the training + test sample, labeled by Gaia designation

cannon_training_set.csv : contains names and labels for 80% of spectra that go into training set
cannon_test_set.csv : contains names and labels for 20% of spectra that go into test set

gaia_rvs_wavelength.fits : RVS spectrum wavelength, which is the same for all Gaia RVS spectra
training_set_flux.fits : array of training set fluxes, in same order as cannon_training_set.csv (faster than loading from flux_data.csv)
training_set_sigma.fits : array of training set flux errors, in same order as cannon_training_set.csv (faster than loading from flux_err_data.csv)

#### this one can probably be deleted
load_cannon_training_set.py	: for old data, can probably be deleted
