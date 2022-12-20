import numpy as np 
import pandas as pd
from astropy.table import Table
import thecannon as tc
import time

# Load the table containing the training set labels
training_set_df = pd.read_csv('./GALAH_csv_files/GALAH_training_set_top5k.csv')
training_set_df_cannon_labels = training_set_df[['teff_gspphot','logg_gspphot', 'mh_gspphot', 'vbroad']]
training_set = Table.from_pandas(training_set_df_cannon_labels) # convert to astropy table

# load the spectra flux + flux errors
f_start = './GALAH_training_set_RVS_spectra/RVS-'
f_end = '.csv'
# flux
print('loading flux for {} spectra'.format(len(training_set_df)))
normalized_flux_nan = np.array([pd.read_csv(f_start+i+f_end).flux.to_numpy() for i in training_set_df.dr3_designation])
normalized_flux = np.nan_to_num(normalized_flux_nan,nan=1) # remove nan values

# inverse variance (gaia reports error=std dev = sqrt(variance))
print('loading flux errors for {} spectra'.format(len(training_set_df)))
normalized_sigma = np.array([pd.read_csv(f_start+i+f_end).flux_error.to_numpy() for i in training_set_df.dr3_designation])
normalized_ivar_nan = 1./(normalized_sigma**2.)# convert to ivar
normalized_ivar = np.nan_to_num(normalized_ivar_nan,nan=np.nanmedian(normalized_ivar_nan))

# Create a vectorizer that defines our model form.
vectorizer = tc.vectorizer.PolynomialVectorizer(('teff_gspphot','logg_gspphot', 'mh_gspphot', 'vbroad'), 2)

# Create the model that will run in parallel using all available cores.
model = tc.CannonModel(training_set, normalized_flux, normalized_ivar,
                       vectorizer=vectorizer)

# train model
start = time.time()
model.train()
end = time.time()
print('finished training cannon model')
print('total time: {} minutes'.format((start-end)/60))
model.write('top5k_cannon_model.model')
print('model written to top5k_cannon_model.model')