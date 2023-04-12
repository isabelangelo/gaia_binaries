import numpy as np 
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import thecannon as tc

# load the labels + flux, ivar for GALAH sample
print('loading GALAH sample')
GALAH_star_labels = pd.read_csv('../process_GALAH_data/GALAH_data_tables/GALAH_stars_filtered.csv')
GALAH_star_flux = pd.read_csv('flux_data.csv')
GALAH_star_flux_err = pd.read_csv('flux_err_data.csv')

# split into training + test sets
np.random.seed(1234)
print('splitting into training + test sets')
GALAH_star_designations = GALAH_star_flux.columns.to_numpy()
n_total = len(GALAH_star_designations)
n_training = int(0.8*n_total)
n_test = n_total - n_training
training_designations = np.random.choice(GALAH_star_designations, size=n_training, replace=False)
test_designations = np.array(list((set(GALAH_star_designations) - set(training_designations))))

# insert tests here
print('checking test + training samples:')
if len(set(GALAH_star_designations) - set(np.concatenate((test_designations, training_designations))))>0:
	print('some stars from sample missing from training + test samples')
elif len(np.intersect1d(training_designations, test_designations))>0:
	print('at least 1 star found in both training + test sample')
elif len(test_designations)+len(training_designations)!=len(GALAH_star_designations):
	print('test + training set dont add up to full length')
else:
	print('tests passed') # or something to that effect

# write training + test set dataframes to .csv files
training_designations_str = ['Gaia DR3 '+i for i in training_designations]
test_designations_str = ['Gaia DR3 '+i for i in test_designations]
training_set_df = GALAH_star_labels[GALAH_star_labels['gaia_designation'].isin(training_designations_str)]
test_set_df = GALAH_star_labels[GALAH_star_labels['gaia_designation'].isin(test_designations_str)]
training_set_df.to_csv('cannon_training_set.csv')
test_set_df.to_csv('cannon_test_set.csv')
print('training set saved to cannon_training_set.csv')
print('training set saved to cannon_test_set.csv')

# write training set flux, ivar to file
training_set_flux = np.array([GALAH_star_flux[designation[9:]].to_numpy() for designation in training_set_df.gaia_designation])
training_set_sigma = np.array([GALAH_star_flux_err[designation[9:]].to_numpy() for designation in training_set_df.gaia_designation])
fits.HDUList([fits.PrimaryHDU(training_set_flux)]).writeto('training_set_flux.fits')
fits.HDUList([fits.PrimaryHDU(training_set_sigma)]).writeto('training_set_sigma.fits')
print('training set fluxes saved to training_set_flux.fits')
print('training set sigma saved to training_set_sigma.fits')

# write test set flux, ivar to file
test_set_flux = np.array([GALAH_star_flux[designation[9:]].to_numpy() for designation in test_set_df.gaia_designation])
test_set_sigma = np.array([GALAH_star_flux_err[designation[9:]].to_numpy() for designation in test_set_df.gaia_designation])
fits.HDUList([fits.PrimaryHDU(test_set_flux)]).writeto('test_set_flux.fits')
fits.HDUList([fits.PrimaryHDU(test_set_sigma)]).writeto('test_set_sigma.fits')
print('training set fluxes saved to test_set_flux.fits')
print('training set sigma saved to test_set_sigma.fits')







