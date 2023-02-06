# to do : leave a bunch out of this for the test step!

import numpy as np 
import pandas as pd
from astropy.table import Table
from astropy.io import fits
import thecannon as tc

# Load the table containing the training set labels
training_set_df = pd.read_csv('./GALAH_csv_files/GALAH_training_set.csv')
training_set_df_cannon_labels = training_set_df[['teff_gspphot','logg_gspphot', 'mh_gspphot', 'vbroad']]
training_set = Table.from_pandas(training_set_df_cannon_labels) # convert to astropy table

normalized_flux_nan = fits.open('./cannon_flux_data.fits')[0].data
normalized_ivar_nan = fits.open('./cannon_ivar_data.fits')[0].data

# set nans to one for now
normalized_flux = np.nan_to_num(normalized_flux_nan,nan=1) # remove nan values
normalized_ivar = np.nan_to_num(normalized_ivar_nan,nan=np.nanmedian(normalized_ivar_nan))

# Create a vectorizer that defines our model form.
vectorizer = tc.vectorizer.PolynomialVectorizer(('teff_gspphot','logg_gspphot', 'mh_gspphot', 'vbroad'), 2)

# Create the model that will run in parallel using all available cores.
model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                       vectorizer=vectorizer)

# train model
model.train()
print('finished training cannon model')
model.write('gaia_RVS_cannon_model.model')
print('model written to gaia_RVS_cannon_model.model')

