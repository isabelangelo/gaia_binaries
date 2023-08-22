"""
trains a cannon model and generates validation plots.
"""
import numpy as np 
from astropy.table import Table
from astropy.io import fits
import thecannon as tc
import pandas as pd

# path to save model files to, 
# should be descriptive of current model to be trained
model_fileroot = 'gaia_rvs_model'

################# define training set + get labels ###################################################
# define training set labels
training_labels = ['galah_teff', 'galah_logg','galah_feh', 'galah_alpha', 'galah_vbroad']

# Load the table containing the training set labels
training_set_table = Table.read('./data/label_dataframes/training_labels.csv', format='csv')
training_set = training_set_table[training_labels]

# normalized_sigma = np.nan_to_num(normalized_sigma_nan, nan=1)
normalized_flux = fits.open('./data/cannon_training_data/training_flux.fits')[0].data
normalized_sigma = fits.open('./data/cannon_training_data/training_sigma.fits')[0].data
normalized_ivar = 1/normalized_sigma**2


# Create a vectorizer that defines our model form.
vectorizer = tc.vectorizer.PolynomialVectorizer(training_labels, 2)

# Create the model that will run in parallel using all available cores.
model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                       vectorizer=vectorizer, regularization=None)

# train model
model_filename = './data/cannon_models/' + model_fileroot + '.model'
model.train()
print('finished training cannon model')
model.write(model_filename, include_training_set_spectra=True)
print('model written to {}'.format(model_filename))

# print training set GALAH label errors
print('average GALAH label uncertainties in training set:')
for label in training_labels:
	print('{}: {}'.format(
		label,
		np.mean(training_set_table[label.replace('_', '_e')])))

