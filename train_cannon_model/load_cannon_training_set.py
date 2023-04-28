import numpy as np 
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import thecannon as tc

# load the labels + flux, ivar for GALAH sample
print('loading GALAH sample')
GALAH_star_labels = pd.read_csv('../process_GALAH_data/GALAH_data_tables/GALAH_stars_filtered.csv')
GALAH_star_flux = pd.read_csv('./data_files/flux_data.csv')
GALAH_star_flux_err = pd.read_csv('./data_files/flux_err_data.csv')

# split into training + test sets
np.random.seed(1234)
print('splitting into training + test sets')
GALAH_star_source_ids = np.array([int(i) for i in GALAH_star_flux.columns.to_numpy()])
n_total = len(GALAH_star_source_ids)
n_training = int(0.8*n_total)
n_test = n_total - n_training
training_source_ids = np.random.choice(GALAH_star_source_ids, size=n_training, replace=False)
test_source_ids = np.array(list((set(GALAH_star_source_ids) - set(training_source_ids))))

# write training + test set dataframes to .csv files
training_set_df = GALAH_star_labels[GALAH_star_labels['source_id'].isin(training_source_ids)]
test_set_df = GALAH_star_labels[GALAH_star_labels['source_id'].isin(test_source_ids)]
training_set_df.to_csv('./data_files/cannon_training_set.csv')
test_set_df.to_csv('./data_files/cannon_test_set.csv')
print('training set saved to ./data_files/cannon_training_set.csv')
print('training set saved to ./data_files/cannon_test_set.csv')


# problem: the fluxes are appended to the array based on training_source_ids, which is random
# but the training set by the original dataframe that they're loaded from
# maybe this would be different if I loaded from training_set_df.source_id

# write training set flux, ivar to file
training_set_flux = np.array([GALAH_star_flux[str(source_id)].to_numpy() for source_id in training_set_df.source_id])
training_set_sigma = np.array([GALAH_star_flux_err[str(source_id)].to_numpy() for source_id in training_set_df.source_id])
fits.HDUList([fits.PrimaryHDU(training_set_flux)]).writeto('./data_files/training_set_flux.fits', overwrite=True)
fits.HDUList([fits.PrimaryHDU(training_set_sigma)]).writeto('./data_files/training_set_sigma.fits', overwrite=True)
print('training set fluxes saved to ./data_files/training_set_flux.fits')
print('training set sigma saved to ./data_files/training_set_sigma.fits')

# write test set flux, ivar to file
test_set_flux = np.array([GALAH_star_flux[str(source_id)].to_numpy() for source_id in test_set_df.source_id])
test_set_sigma = np.array([GALAH_star_flux_err[str(source_id)].to_numpy() for source_id in test_set_df.source_id])
fits.HDUList([fits.PrimaryHDU(test_set_flux)]).writeto('./data_files/test_set_flux.fits', overwrite=True)
fits.HDUList([fits.PrimaryHDU(test_set_sigma)]).writeto('./data_files/test_set_sigma.fits', overwrite=True)
print('training set fluxes saved to ./data_files/test_set_flux.fits')
print('training set sigma saved to ./data_files/test_set_sigma.fits')






